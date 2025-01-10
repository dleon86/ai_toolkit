import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import cv2
from PIL import Image
import os
import json
from datetime import datetime
import threading
import sys
from queue import Queue
import time
import pandas as pd
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights

# Load ResNet model with the latest API
weights = ResNet50_Weights.DEFAULT
resnet_model = resnet50(weights=weights)
resnet_model.eval()

# Load ImageNet class labels
with open("imagenet_classes.txt") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    ),
])

class TimeoutError(Exception):
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=60):
    """Run a function with a timeout"""
    result = Queue()
    
    def worker():
        try:
            result.put(('success', func(*args, **kwargs)))
        except Exception as e:
            result.put(('error', e))
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        raise TimeoutError("Processing took too long!")
    
    status, value = result.get()
    if status == 'error':
        raise value
    return value

def segment_image(image_path, output_dir=None):
    """Segment an image and save the results"""
    try:
        # Initialize the model
        mask_generator = load_sam2()
        
        # Load local image
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
            
        image = Image.open(image_path)
        print(f"Loaded image from: {image_path}")
        
        # Convert image to numpy array
        if isinstance(image, dict) and 'array' in image:
            image = image['array']
        image = np.array(image)
        
        # Resize image if too large
        max_size = 800
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            print(f"Resized image to {new_w}x{new_h}")
        
        print("Processing image...")
        print("Image shape:", image.shape if hasattr(image, 'shape') else "No shape available")
        
        # Generate masks with timeout
        masks = run_with_timeout(
            mask_generator.generate,
            args=(image,),
            timeout_duration=300
        )
        print(f"Generated {len(masks)} segments")
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = "saved_segments"
        save_dir = os.path.join(output_dir, f"segments_{timestamp}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save segments and metadata
        save_segments(image, masks, save_dir, timestamp)
        
        print(f"\nSegmentation complete! Results saved in: {save_dir}")
        print(f"Total segments: {len(masks)}")
        
        return save_dir
        
    except Exception as e:
        print(f"\nAn error occurred during segmentation: {str(e)}")
        return None

def classify_segments(segments_dir):
    """Classify segments in a directory using ResNet and save results to CSV"""
    try:
        if not os.path.exists(segments_dir):
            print(f"Error: Directory not found: {segments_dir}")
            return None
            
        print(f"Processing segments in: {segments_dir}")
        
        # Load metadata if it exists
        metadata_files = [f for f in os.listdir(segments_dir) if f.startswith('segments_metadata_') and f.endswith('.json')]
        if metadata_files:
            metadata_path = os.path.join(segments_dir, metadata_files[0])
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"Loaded metadata from: {metadata_path}")
        else:
            # If no metadata, just process all PNG files
            segment_files = [f for f in os.listdir(segments_dir) if f.startswith('masked_') and f.endswith('.png')]
            metadata = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "segments": [{"segment_id": i, "file_path": os.path.join(segments_dir, f)} 
                            for i, f in enumerate(segment_files)]
            }
        
        results = []
        print("\nClassifying segments...")
        for segment in metadata["segments"]:
            segment_path = segment["file_path"]
            print(f"Processing segment ID {segment['segment_id']}: {segment_path}")
            if not os.path.exists(segment_path):
                print(f"Warning: Segment file not found: {segment_path}")
                continue
                
            # Open and preprocess the image
            try:
                img = Image.open(segment_path).convert('RGB')
                print(f"Image size: {img.size}, Mode: {img.mode}")
                img_t = preprocess(img)
                batch_t = torch.unsqueeze(img_t, 0)
                
                # Run ResNet classification
                with torch.no_grad():
                    out = resnet_model(batch_t)
                
                # Get top 5 predictions
                probabilities = torch.nn.functional.softmax(out[0], dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 5)
                top5_classes = [imagenet_classes[catid] for catid in top5_catid.cpu().numpy()]
                top5_conf = top5_prob.cpu().numpy().tolist()
                
                # Get top prediction
                top_conf = top5_conf[0]
                class_name = top5_classes[0]
                
                # Log classification details
                print(f"Segment ID {segment['segment_id']} classification:")
                print(f"  Predicted Class: {class_name} ({top_conf:.2f})")
                print(f"  Top 5 Classes: {top5_classes}")
                print(f"  Top 5 Confidences: {top5_conf}")
                
                # Create result dictionary
                result = {
                    "segment_id": segment["segment_id"],
                    "file_path": segment_path,
                    "predicted_class": class_name,
                    "confidence": top_conf,
                    "top5_classes": top5_classes,
                    "top5_confidences": top5_conf
                }
                
                # Add metadata if available
                if "area" in segment:
                    result.update({
                        "area": segment["area"],
                        "bbox": segment["bbox"],
                        "predicted_iou": segment["predicted_iou"],
                        "stability_score": segment["stability_score"]
                    })
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing segment {segment['segment_id']}: {e}")
                continue
        
        # Save results to CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(segments_dir, f"classifications_{metadata['timestamp']}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved classifications to: {csv_path}")
        
        return results
        
    except Exception as e:
        print(f"\nAn error occurred during classification: {str(e)}")
        return None

def save_segments(image, masks, save_dir, timestamp):
    """Save segmented images and metadata"""
    metadata = {
        "timestamp": timestamp,
        "segments": []
    }
    for i, mask in enumerate(masks):
        # Apply mask to the image
        masked_image = apply_mask(image, mask)
        segment_filename = f"masked_{timestamp}_segment{i}.png"
        segment_path = os.path.join(save_dir, segment_filename)
        Image.fromarray(masked_image).save(segment_path)
        
        # Extract bounding box and area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = [x, y, w, h]
            area = cv2.contourArea(cnt)
        else:
            bbox = [0, 0, 0, 0]
            area = 0.0
        
        # Dummy values for predicted_iou and stability_score
        predicted_iou = 0.0
        stability_score = 0.0
        
        metadata["segments"].append({
            "segment_id": i,
            "file_path": segment_path,
            "area": area,
            "bbox": bbox,
            "predicted_iou": predicted_iou,
            "stability_score": stability_score
        })
    
    # Save metadata
    metadata_path = os.path.join(save_dir, f"segments_metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to: {metadata_path}")

def apply_mask(image, mask):
    """Apply binary mask to the image"""
    masked_image = np.copy(image)
    masked_image[mask == 0] = 0
    return masked_image

def main(image_path=None, segments_dir=None):
    """
    Main function that can either:
    1. Process a new image (segment + classify)
    2. Classify existing segments
    
    Args:
        image_path: Path to image to segment and classify
        segments_dir: Path to directory containing existing segments
    """
    try:
        if segments_dir:
            # Only run classification on existing segments
            print("Using existing segments...")
            classify_segments(segments_dir)
        elif image_path:
            # Run full pipeline
            print("Running full segmentation pipeline...")
            segments_dir = segment_image(image_path)
            if segments_dir:
                classify_segments(segments_dir)
        else:
            print("Error: Must provide either image_path or segments_dir")
            return
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)

def classify_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            out = resnet_model(batch_t)
        
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        top5_classes = [imagenet_classes[catid] for catid in top5_catid.cpu().numpy()]
        top5_conf = top5_prob.cpu().numpy().tolist()
        
        return top5_classes, top5_conf
    except Exception as e:
        print(f"Error classifying image {image_path}: {e}")
        return [], []

# Test with a few images
test_images = [
    "path/to/segment0.png",
    "path/to/segment1.png",
    "path/to/segment2.png"
]

for img_path in test_images:
    classes, confidences = classify_image(img_path)
    print(f"Image: {img_path}")
    print(f"  Top 5 Classes: {classes}")
    print(f"  Top 5 Confidences: {confidences}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Segment and classify image regions')
    parser.add_argument('--image', help='Path to image to segment and classify')
    parser.add_argument('--segments', help='Path to existing segments directory')
    
    args = parser.parse_args()
    
    if not args.image and not args.segments:
        # Use default example
        segments_dir = "./saved_segments/segments_20250109_210734"
        print(f"No input provided, using example directory: {segments_dir}")
        main(segments_dir=segments_dir)
    else:
        main(image_path=args.image, segments_dir=args.segments) 