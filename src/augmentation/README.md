# NEU Metal Surface Defects YOLO Data Augmentation

## Purpose of the Augmentation Module
This module provides a robust data augmentation pipeline for the NEU Metal Surface Defects dataset. It applies various geometric and color-based transformations to the images while properly maintaining and adjusting the corresponding YOLOv8 bounding box annotations. 

## Why Augmentation is Used
Data augmentation is a critical technique in deep learning to artificially expand the diversity of the training dataset. For industrial defect detection, this helps the YOLOv8 model generalize better by learning to recognize defects under varying lighting conditions, orientations, and slight camera noises without needing to capture tens of thousands of new physical samples.

## Augmentations Selected and Why
We use `Albumentations` to apply realistic and appropriate transformations for metal surface defects:
- **Small Rotation (up to 15°)**: Industrial cameras might have slight misalignment, but not extreme rotations (e.g., 90° or 180° for strongly directional textures like scratches).
- **Brightness & Contrast Variation**: Simulates different lighting conditions in a factory environment.
- **Gaussian Noise**: Simulates camera sensor noise, particularly in low-light factory settings.
- **Mild Blur**: Simulates slight motion blur or out-of-focus camera shots on a moving production line.
- **Horizontal & Vertical Flips**: Defects like crazing, pitted surfaces, and patches often have no inherent orientation and can appear in any flipped configuration.

## YOLO Bounding-Box Format
The script requires annotations in the YOLO format:
`class_id x_center y_center width height`
where `x_center`, `y_center`, `width`, and `height` are normalized to the range `[0.0, 1.0]` relative to the image dimensions.
Classes for this dataset are `0: crazing`, `1: inclusion`, `2: patches`, `3: pitted_surface`, `4: rolled-in_scale`, `5: scratches`.

## Input/Output Directory Structure
The script expects the standard YOLO dataset structure:
```text
Dataset/
├── images/
│   ├── train/
│   └── validation/
└── labels/
    ├── train/
    └── validation/
```
By default, the script reads from `./Dataset` and outputs the augmented dataset to `./Dataset_Augmented` while preserving the exact same directory layout.

## Configurable Augmentation Behavior
To prevent blindly generating thousands of augmented copies that could bloat the dataset or lead to overfitting, the augmentation is controlled by two configurable parameters:
- `--prob`: The probability (0.0 to 1.0) of applying augmentation to any given training image.
- `--count`: The number of augmented copies generated per selected image.

This probability-based approach ensures a controlled expansion of the dataset, keeping training times reasonable while injecting sufficient variance.

## Validation Performed
The module performs several validation checks on bounding boxes before and after augmentation:
- Ensures class IDs are valid integers between 0 and 5.
- Verifies that all coordinates (x, y, w, h) are finite numbers.
- Checks that `width` and `height` are strictly positive (and <= 1.0).
- Confirms normalized coordinates remain within the valid [0.0, 1.0] range.
- Ensures every augmented image successfully written has a corresponding label file (even if it's empty because bounding boxes were dropped by cropping/transformations).
- Any invalid bounding boxes are skipped and warnings are logged.

## ⚠️ Important Notes on Validation Data
**Validation data is NEVER augmented.**
Applying augmentation to validation data corrupts the validation metrics, as it no longer represents the real-world distribution of unseen data. The script explicitly only augments the `images/train` and `labels/train` directories. The `validation` directories are copied to the output folder completely unchanged.

*Note: Final validation of this augmentation pipeline should be performed on the real NEU-DET dataset once it is available in the environment.*

## Installation

Ensure you have a Python environment (e.g. >= 3.8). Install the required dependencies using `pip`:
```bash
pip install albumentations opencv-python numpy
```

## Usage

Run the augmentation with default settings (reads from `./Dataset`, writes to `./Dataset_Augmented`, 50% probability, 1 copy):
```bash
python src/augmentation/augmentation.py
```

Override default parameters:
```bash
python src/augmentation/augmentation.py --input path/to/input --output path/to/output --prob 0.8 --count 2
```

## Running the Synthetic Test
Because the actual NEU dataset might not be locally available, a synthetic test mode is included. This generates a mock image with a bounding box, passes it through the Albumentations pipeline, and verifies that the bounding box is successfully transformed without requiring the real dataset.

Run the test:
```bash
python src/augmentation/augmentation.py --test
```
This will output `synthetic_original.jpg` and `synthetic_augmented.jpg` in the current directory and log the original and augmented bounding box coordinates to the console.
