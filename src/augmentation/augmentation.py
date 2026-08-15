import argparse
import os
import glob
import cv2
import albumentations as A
import logging
import numpy as np
from pathlib import Path
import random
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_yolo_bbox(bbox, class_id):
    """
    Validates a single YOLO bounding box.
    Returns True if valid, False otherwise.
    bbox format: (x_center, y_center, width, height) - normalized
    """
    try:
        class_id = int(class_id)
        if not (0 <= class_id <= 5):
            return False
    except ValueError:
        return False
        
    if len(bbox) != 4:
        return False
        
    x_c, y_c, w, h = bbox
    
    if not all(np.isfinite([x_c, y_c, w, h])):
        return False
        
    if not (0.0 < w <= 1.0) or not (0.0 < h <= 1.0):
        return False
        
    if not (0.0 <= x_c <= 1.0) or not (0.0 <= y_c <= 1.0):
        return False

    return True

def get_augmentation_pipeline():
    """
    Returns an Albumentations composition for industrial metal surface defects.
    - Small rotation (up to 15 degrees)
    - Brightness/Contrast variation
    - Gaussian Noise
    - Mild Blur
    - Flips
    """
    return A.Compose([
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))

def parse_yolo_label(label_path):
    """
    Parses a YOLO label file.
    Returns a list of bboxes, a list of class labels, and the number of invalid annotations.
    """
    bboxes = []
    class_labels = []
    invalid_count = 0
    if not os.path.exists(label_path):
        return bboxes, class_labels, invalid_count
        
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = parts[0]
                try:
                    bbox = [float(p) for p in parts[1:5]]
                    if validate_yolo_bbox(bbox, class_id):
                        bboxes.append(bbox)
                        class_labels.append(int(class_id))
                    else:
                        logging.warning(f"Invalid bounding box in {label_path}: {line.strip()}")
                        invalid_count += 1
                except ValueError:
                    logging.warning(f"Malformed bounding box coordinates in {label_path}: {line.strip()}")
                    invalid_count += 1
            elif line.strip():
                logging.warning(f"Malformed bounding box line in {label_path}: {line.strip()}")
                invalid_count += 1
    return bboxes, class_labels, invalid_count

def save_yolo_label(label_path, bboxes, class_labels):
    """
    Saves bounding boxes to a YOLO label file.
    """
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            if validate_yolo_bbox(bbox, class_id):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

def copy_validation_data(input_dir, output_dir):
    """
    Copies validation data unchanged from input_dir to output_dir.
    """
    val_dirs = [
        (Path(input_dir) / 'images' / 'validation', Path(output_dir) / 'images' / 'validation'),
        (Path(input_dir) / 'labels' / 'validation', Path(output_dir) / 'labels' / 'validation')
    ]
    for src_dir, dst_dir in val_dirs:
        if src_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            for item in src_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst_dir / item.name)
            logging.info(f"Copied validation data from {src_dir} to {dst_dir}")
        else:
            logging.warning(f"Validation directory not found: {src_dir}")

def augment_dataset(input_dir, output_dir, prob=0.5, max_aug_per_image=1):
    """
    Augments the YOLO dataset located in input_dir and saves to output_dir.
    Applies augmentations only to 'train' split.
    """
    img_train_dir = Path(input_dir) / 'images' / 'train'
    lbl_train_dir = Path(input_dir) / 'labels' / 'train'
    
    if not img_train_dir.exists():
        logging.error(f"Training images directory not found: {img_train_dir}")
        return
        
    out_img_train_dir = Path(output_dir) / 'images' / 'train'
    out_lbl_train_dir = Path(output_dir) / 'labels' / 'train'
    
    out_img_train_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_train_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = get_augmentation_pipeline()
    image_paths = glob.glob(str(img_train_dir / '*.jpg')) + glob.glob(str(img_train_dir / '*.png'))
    
    logging.info(f"Found {len(image_paths)} images in {img_train_dir}")
    
    # Copy validation data without modifying
    copy_validation_data(input_dir, output_dir)
    
    total_invalid_annotations = 0
    
    for img_path in image_paths:
        img_path = Path(img_path)
        label_path = lbl_train_dir / f"{img_path.stem}.txt"
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            logging.error(f"Failed to read image: {img_path}")
            continue
            
        # Parse labels
        bboxes, class_labels, invalid_count = parse_yolo_label(label_path)
        total_invalid_annotations += invalid_count
        
        # We always copy the original image to output first to keep the original train data
        out_img_path = out_img_train_dir / img_path.name
        out_lbl_path = out_lbl_train_dir / label_path.name
        
        cv2.imwrite(str(out_img_path), image)
        
        # Preserve the original annotation file exactly as it was
        if label_path.exists():
            shutil.copy2(label_path, out_lbl_path)
        else:
            open(out_lbl_path, 'w').close()
            
        # Skip augmentation if this image has invalid annotations
        if invalid_count > 0:
            logging.warning(f"Skipping augmentation for {img_path.name} due to invalid annotations.")
            continue
            
        # Probability based augmentation
        if random.random() > prob:
            continue
            
        for i in range(max_aug_per_image):
            try:
                # Albumentations expects RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply transformation
                if bboxes:
                    transformed = pipeline(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_class_labels = transformed['class_labels']
                else:
                    transformed = pipeline(image=image_rgb)
                    aug_image = transformed['image']
                    aug_bboxes = []
                    aug_class_labels = []
                    
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                
                aug_img_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                aug_lbl_name = f"{img_path.stem}_aug_{i}.txt"
                
                out_aug_img_path = out_img_train_dir / aug_img_name
                out_aug_lbl_path = out_lbl_train_dir / aug_lbl_name
                
                # Save augmented image and label
                cv2.imwrite(str(out_aug_img_path), aug_image_bgr)
                save_yolo_label(out_aug_lbl_path, aug_bboxes, aug_class_labels)
                
                if not out_aug_lbl_path.exists():
                    open(out_aug_lbl_path, 'w').close()
                
            except Exception as e:
                logging.error(f"Error augmenting image {img_path}: {e}")
                
    if total_invalid_annotations > 0:
        logging.warning(f"Total invalid annotations skipped during processing: {total_invalid_annotations}")
    else:
        logging.info("No invalid annotations found during processing.")

def create_synthetic_test():
    """
    Creates a small synthetic image and bounding box, and tests the augmentation pipeline
    by applying a deterministic horizontal flip and mathematically verifying the result.
    """
    logging.info("Running deterministic synthetic test...")
    
    # Create a synthetic image (e.g., a noisy gray background with a distinct rectangle)
    image = np.ones((200, 200, 3), dtype=np.uint8) * 128
    
    # Draw an asymmetrical defect
    cv2.rectangle(image, (50, 50), (100, 100), (0, 0, 255), -1)
    
    # Bbox: x_c, y_c, w, h in normalized coordinates
    # x_c = (50 + 100) / 2 / 200 = 75 / 200 = 0.375
    # y_c = (50 + 100) / 2 / 200 = 75 / 200 = 0.375
    # w = 50 / 200 = 0.25
    # h = 50 / 200 = 0.25
    original_bbox = [0.375, 0.375, 0.25, 0.25]
    bboxes = [original_bbox]
    class_labels = [0] # e.g., crazing
    
    # Use a deterministic pipeline for verification instead of the random one
    deterministic_pipeline = A.Compose([
        A.HorizontalFlip(p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    try:
        # Save original test image for visual check
        cv2.imwrite("synthetic_original.jpg", image)
        
        # Apply deterministic transformation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = deterministic_pipeline(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']
        
        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("synthetic_augmented.jpg", aug_img_bgr)
        
        logging.info(f"Original bbox: {original_bbox}")
        
        if not aug_bboxes:
            raise ValueError("Augmented bbox was unexpectedly removed.")
            
        aug_bbox = aug_bboxes[0]
        logging.info(f"Augmented bbox: {aug_bbox}")
        
        # Mathematically verify a horizontal flip
        # Expected: x_center becomes 1.0 - x_center, other dimensions remain same
        expected_bbox = [1.0 - original_bbox[0], original_bbox[1], original_bbox[2], original_bbox[3]]
        
        # Allow small floating point tolerance
        if not np.allclose(aug_bbox, expected_bbox, atol=1e-5):
            raise ValueError(f"Mathematical verification failed! Expected {expected_bbox}, got {aug_bbox}")
            
        logging.info("Deterministic synthetic test passed. Bounding box transformed correctly.")
        logging.info("Check synthetic_original.jpg and synthetic_augmented.jpg")
        
    except Exception as e:
        logging.error(f"Synthetic test failed: {e}")
        import sys
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Data Augmentation for NEU Metal Surface Defects")
    parser.add_argument('--input', type=str, default='./Dataset', help="Input dataset directory")
    parser.add_argument('--output', type=str, default='./Dataset_Augmented', help="Output dataset directory")
    parser.add_argument('--prob', type=float, default=0.5, help="Probability of augmenting an image")
    parser.add_argument('--count', type=int, default=1, help="Number of augmented copies per selected image")
    parser.add_argument('--test', action='store_true', help="Run synthetic test without dataset")
    
    args = parser.parse_args()
    
    if args.test:
        create_synthetic_test()
    else:
        logging.info(f"Starting augmentation from {args.input} to {args.output}")
        augment_dataset(args.input, args.output, prob=args.prob, max_aug_per_image=args.count)
        logging.info("Augmentation completed.")

if __name__ == '__main__':
    main()
