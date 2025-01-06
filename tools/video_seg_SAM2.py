import torch
from sam2.build_sam import build_sam2_video_predictor
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from datetime import datetime
import imageio
import io
from IPython.display import Image as IPyImage
from IPython.display import display
import matplotlib.animation as animation

# Initialize global variables for storing points
points = []
labels = []
current_masks = []
save_dir = "saved_video_segments"

def load_sam2_video():
    # Get the root directory path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load SAM2 video model
    checkpoint = os.path.join(root_dir, "external", "sam2", "checkpoints", "sam2.1_hiera_tiny.pt")
    model_cfg = os.path.join(root_dir, "external", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_t.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    return predictor

def extract_frames(video_path, output_dir, frame_rate=1):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the original frame rate of the video
    original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_frame_rate / frame_rate)
    if frame_interval == 0:
        frame_interval = 1  # Ensure at least every frame is captured if frame_rate > original_frame_rate

    frame_count = 0
    saved_frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every nth frame based on the desired frame rate
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_frame_count} frames to {output_dir}")

def segment_video(predictor, video_dir):
    # Scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    # Extract the numeric part of the filename for sorting
    frame_names.sort(key=lambda p: int(p.split('.')[0]))
    
    if not frame_names:
        print("No frames found in the directory.")
        return {}, []

    # Load images in sorted order
    images = [cv2.imread(os.path.join(video_dir, fn)) for fn in frame_names]
    video_height, video_width = images[0].shape[:2]
    
    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_dir)
    
    # Example: Segment & track one object
    ann_frame_idx = 0
    ann_obj_id = 1
    points = np.array([[210, 350]], dtype=np.float32)  # Placeholder, will be updated interactively
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    
    # Propagate the prompts to get the masklet across the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    return video_segments, frame_names

def show_video_masks(video_segments, frame_names, video_dir):
    vis_frame_stride = 4
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"Frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # Squeeze the mask to remove any singleton dimensions
            out_mask = np.squeeze(out_mask)
            plt.imshow(out_mask, alpha=0.5)
        plt.show()

def get_user_points(image):
    """
    Display the image and allow the user to click points to define the object to segment.
    Returns the list of points and their labels.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title("Click on the image to select points for segmentation.\nClose the window when done.")
    plt.axis('off')
    
    clicked_points = []
    
    def onclick(event):
        if event.inaxes != ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        clicked_points.append([x, y])
        ax.plot(x, y, 'r*', markersize=15)
        fig.canvas.draw()
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    if not clicked_points:
        print("No points were clicked. Exiting segmentation.")
        exit(0)
    
    # For simplicity, assign all clicks as foreground points
    clicked_labels = [1] * len(clicked_points)
    return np.array(clicked_points, dtype=np.float32), np.array(clicked_labels, dtype=np.int32)

def initialize_segmentation(predictor, video_dir):
    # Get the first frame
    first_frame_path = os.path.join(video_dir, sorted(os.listdir(video_dir))[0])
    first_image = Image.open(first_frame_path)
    first_image = np.array(first_image)
    
    # Get user points
    point_coords, point_labels = get_user_points(first_image)
    
    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_dir)
    
    # Add user points to the predictor
    ann_frame_idx = 0
    ann_obj_id = 1
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=point_coords,
        labels=point_labels,
    )
    
    # Propagate the prompts to get the masklet across the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    return video_segments, sorted(os.listdir(video_dir))

def ensure_save_dir(video_name):
    """Create the save directory if it doesn't exist"""
    save_dir = os.path.join("saved_segments", video_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    return save_dir

def save_segments(video_dir, video_segments, frame_names):
    """Save the masked and unmasked regions with transparency"""
    if not video_segments:
        print("No segments to save!")
        return
    
    # Extract video name from directory path
    video_name = os.path.basename(video_dir)
    save_dir = ensure_save_dir(video_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each frame
    for frame_idx, masks_dict in video_segments.items():
        # Load original frame
        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        image = Image.open(frame_path)
        image = np.array(image)
        
        # Process each mask in the frame
        for obj_id, mask in masks_dict.items():
            # Ensure mask is 2D
            bool_mask = np.squeeze(mask).astype(bool)
            
            # Create masked image (only the selected region)
            masked_image = image.copy()
            # Add alpha channel
            alpha = np.zeros_like(masked_image[..., 0])
            alpha[bool_mask] = 255  # Set alpha to fully opaque for masked regions
            masked_rgba = np.dstack((masked_image, alpha))
            
            # Create unmasked image (everything except the selected region)
            unmasked_image = image.copy()
            # Add alpha channel
            alpha = np.zeros_like(unmasked_image[..., 0])
            alpha[~bool_mask] = 255  # Set alpha to fully opaque for unmasked regions
            unmasked_rgba = np.dstack((unmasked_image, alpha))
            
            # Save masked region
            masked_path = os.path.join(save_dir, f"masked_{timestamp}_frame{frame_idx:04d}_obj{obj_id}.png")
            Image.fromarray(masked_rgba.astype('uint8'), 'RGBA').save(masked_path)
            
            # Save unmasked region
            unmasked_path = os.path.join(save_dir, f"unmasked_{timestamp}_frame{frame_idx:04d}_obj{obj_id}.png")
            Image.fromarray(unmasked_rgba.astype('uint8'), 'RGBA').save(unmasked_path)
    
    print(f"Saved segments to: {save_dir}")

def create_gif(video_dir, video_segments, frame_names, timestamp):
    """Create a GIF from the masked frames"""
    # Prepare frames for GIF
    gif_frames = []
    
    # Process each frame
    for frame_idx, masks_dict in video_segments.items():
        # Load original frame
        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        image = Image.open(frame_path)
        image = np.array(image)
        
        # Create figure for this frame
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        
        # Add all masks for this frame
        for obj_id, mask in masks_dict.items():
            mask = np.squeeze(mask)
            ax.imshow(mask, alpha=0.5, cmap='jet')
        
        ax.axis('off')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        gif_frames.append(Image.open(buf))
        plt.close()
    
    # Save the GIF
    video_name = os.path.basename(video_dir)
    save_dir = os.path.join("saved_segments", video_name)
    gif_path = os.path.join(save_dir, f"segmentation_{timestamp}.gif")
    
    # Save with a reasonable duration per frame (e.g., 0.1 seconds per frame)
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=100,  # milliseconds per frame
        loop=0
    )
    
    print(f"Saved GIF to: {gif_path}")
    return gif_path

def display_gif(gif_path):
    """Display the GIF using matplotlib animation"""
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frames.append(np.array(gif.copy()))
            gif.seek(len(frames))
    except EOFError:
        pass
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('off')
    
    # Create animation
    ani = animation.ArtistAnimation(
        fig, 
        [[ax.imshow(frame)] for frame in frames],
        interval=100,  # milliseconds per frame
        blit=True,
        repeat=True
    )
    
    plt.show()

def main():
    video_path = "./external/sam2/notebooks/videos/acrobats1.mp4"
    output_dir = "./external/sam2/notebooks/videos/acrobats1"
    
    # Extract frames at lower frame rate (e.g., 1 frame per second)
    extract_frames(video_path, output_dir, frame_rate=20)  # Adjust frame_rate as needed
    
    # Initialize the model
    predictor = load_sam2_video()
    
    # Initialize segmentation with user input on the first frame
    video_segments, frame_names = initialize_segmentation(predictor, output_dir)
    
    # Get timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Show the results
    show_video_masks(video_segments, frame_names, output_dir)
    
    # Save the segments
    save_segments(output_dir, video_segments, frame_names)
    
    # Create and display GIF
    gif_path = create_gif(output_dir, video_segments, frame_names, timestamp)
    display_gif(gif_path)

if __name__ == "__main__":
    main()