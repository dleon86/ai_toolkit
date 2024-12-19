import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import os
from datetime import datetime

# Initialize global variables for storing points
points = []
labels = []
current_masks = []
save_dir = "saved_segments"

def load_sam2():
    # Get the root directory path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load SAM2 model
    checkpoint = os.path.join(root_dir, "external", "sam2", "checkpoints", "sam2.1_hiera_large.pt")
    model_cfg = os.path.join(root_dir, "external", "sam2", "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def segment_image(predictor, image, point_coords=None, point_labels=None):
    # Convert PIL Image to numpy array if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Ensure image is RGB
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA
        image = image[:,:,:3]
    
    # Set the image in the predictor
    predictor.set_image(image)
    
    # Use torch's automatic mixed precision for faster inference
    with torch.inference_mode():
        if point_coords is None or len(point_coords) == 0:
            raise NotImplementedError("No points provided for segmentation.")
        else:
            # Generate mask based on point prompts
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
    
    return masks, scores, image

def show_masks(ax, image, masks, scores, points, labels):
    # Clear previous masks
    ax.imshow(image)
    
    # Overlay masks
    for mask in masks:
        ax.imshow(mask, alpha=0.5)
    
    # Redraw points
    for point in points:
        ax.plot(point[0], point[1], 'r*', markersize=15)
    
    ax.figure.canvas.draw()

def onclick(event, ax, fig, predictor, image):
    if event.inaxes != ax:
        return
    
    x, y = int(event.xdata), int(event.ydata)
    print(f"Clicked at: ({x}, {y})")
    
    # For simplicity, we'll consider all clicks as foreground points
    points.append([x, y])
    labels.append(1)
    
    # Update the plot with the new point
    ax.plot(x, y, 'r*', markersize=15)
    fig.canvas.draw()
    
    # Segment the image with the current points
    point_coords = np.array(points)
    point_labels_np = np.array(labels)
    
    masks, scores, processed_image = segment_image(predictor, image, point_coords, point_labels_np)
    
    # Store current masks for potential future use
    global current_masks
    current_masks = masks
    
    # Show the updated masks
    show_masks(ax, processed_image, masks, scores, points, labels)

def clear(event, ax, fig, image):
    global points, labels, current_masks
    points = []
    labels = []
    current_masks = []
    ax.imshow(image)
    ax.set_title("Click on the image to select points for segmentation")
    fig.canvas.draw()

def ensure_save_dir():
    """Create the save directory if it doesn't exist"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

def save_segments(event, image, masks):
    """Save the masked and unmasked regions with transparency"""
    if masks is None or not isinstance(masks, np.ndarray) or masks.size == 0:
        print("No masks to save!")
        return
    
    ensure_save_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert image to PIL format if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Save original image
    image_path = os.path.join(save_dir, f"original_{timestamp}.png")
    image.save(image_path)
    print(f"Saved original image to: {image_path}")
    
    # Process each mask
    for i, mask in enumerate(masks):
        # Convert mask to boolean array
        bool_mask = mask.astype(bool)
        
        # Create masked image (only the selected region)
        masked_image = np.array(image).copy()
        # Add alpha channel
        alpha = np.zeros_like(masked_image[..., 0])
        alpha[bool_mask] = 255  # Set alpha to fully opaque for masked regions
        masked_rgba = np.dstack((masked_image, alpha))
        
        # Create unmasked image (everything except the selected region)
        unmasked_image = np.array(image).copy()
        # Add alpha channel
        alpha = np.zeros_like(unmasked_image[..., 0])
        alpha[~bool_mask] = 255  # Set alpha to fully opaque for unmasked regions
        unmasked_rgba = np.dstack((unmasked_image, alpha))
        
        # Save masked region
        masked_path = os.path.join(save_dir, f"masked_{timestamp}_segment{i}.png")
        Image.fromarray(masked_rgba.astype('uint8'), 'RGBA').save(masked_path)
        print(f"Saved masked region to: {masked_path}")
        
        # Save unmasked region
        unmasked_path = os.path.join(save_dir, f"unmasked_{timestamp}_segment{i}.png")
        Image.fromarray(unmasked_rgba.astype('uint8'), 'RGBA').save(unmasked_path)
        print(f"Saved unmasked region to: {unmasked_path}")

def main(image_number=0):
    # Initialize the model
    predictor = load_sam2()
    
    # Load BSD100 dataset with caching
    dataset = load_dataset("eugenesiow/BSD100", trust_remote_code=True, cache_dir="./cached_datasets")
    
    # Get the first high-resolution image
    first_item = dataset["validation"][image_number]
    image_address = first_item["hr"]  # Using high-resolution image
    image = Image.open(image_address)
    
    # Convert image to numpy array
    if isinstance(image, dict) and 'array' in image:
        image = image['array']
    image = np.array(image)
    
    print("Image type:", type(image))
    print("Image shape:", image.shape if hasattr(image, 'shape') else "No shape available")
    
    # Setup Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title("Left Click: Foreground | Right Click: Background")
    
    # Add Clear button
    ax_clear = plt.axes([0.7, 0.01, 0.1, 0.05])  # Moved left to make room for Save button
    btn_clear = Button(ax_clear, 'Clear')
    btn_clear.on_clicked(lambda event: clear(event, ax, fig, image))
    
    # Add Save button
    ax_save = plt.axes([0.85, 0.01, 0.1, 0.05])
    btn_save = Button(ax_save, 'Save')
    btn_save.on_clicked(lambda event: save_segments(event, image, current_masks))
    
    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', 
                                lambda event: onclick(event, ax, fig, predictor, image))
    
    plt.show()

if __name__ == "__main__":
    main(image_number=66) 