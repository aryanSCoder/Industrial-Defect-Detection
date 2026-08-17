
# YOLOv8 Training Configuration - Week 2

## Setup Instructions

### Prerequisites
pip install ultralytics torch opencv-python pandas matplotlib

### Dataset Setup
Ensure your data.yaml is configured:

path: C:\Users\revat\Downloads\Neu zaalima dataset\NEU-DET
train: train/images
val: validation/images

nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']

### Running Training

Local Machine:
python yolov8_train_neu.py

Google Colab:
from yolov8_train_neu import NEUYOLOTrainer
trainer = NEUYOLOTrainer('data.yaml')
results = trainer.train(epochs=50, batch_size=16)

## Training Parameters

- Model: YOLOv8 Nano (lightweight, fast training)
- Epochs: 50 (adjustable)
- Batch Size: 16 (adjustable based on GPU memory)
- Image Size: 640x640
- Optimizer: SGD
- Learning Rate: Auto-calculated
- Early Stopping: Patience=10 epochs

## Augmentation Applied

- HSV: H=0.015, S=0.7, V=0.4
- Rotation: +/- 10 degrees
- Translation: 10%
- Scale: 50%
- Flip: 50% horizontal & vertical
- Mosaic: 100%

## Output

Training results saved to: ./NEU_Detection/run1/

Contains:
- weights/ - Trained model weights
- results.png - Training metrics plots
- F1_curve.png - F1 score curve
- confusion_matrix.png - Confusion matrix

## Metrics Tracked

- mAP@0.5 - Mean Average Precision at IoU 0.5
- mAP@0.5:0.95 - Mean Average Precision at IoU 0.5:0.95
- Precision
- Recall
- F1 Score
- Loss (training & validation)

## Next Steps

1. Train on full dataset
2. Analyze results and metrics
3. Fine-tune hyperparameters if needed
4. Export model to ONNX/TensorRT for deployment
