"""
YOLOv8 Training Script for NEU Metal Surface Defects Detection
Week 2: Model Training & Evaluation
Author: Chaitanya (Member 1)
Matches TRAINING_CONFIG.md specifications
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path

class NEUYOLOTrainer:
    def __init__(self, dataset_yaml_path, project_name="NEU_Detection"):
        """
        Initialize YOLO trainer for NEU dataset
        
        Args:
            dataset_yaml_path: Path to data.yaml file
            project_name: Name of the project for saving results
        """
        self.dataset_yaml = dataset_yaml_path
        self.project_name = project_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Using device: {self.device}")
        
        # Initialize YOLOv8 Nano model (lightweight, fast training)
        self.model = YOLO('yolov8n.pt')
        print(f"✅ YOLOv8 Nano model initialized")
        
    def train(self, epochs=50, batch_size=16, imgsz=640, patience=10):
        """
        Train YOLOv8 model on NEU dataset with augmentation
        
        Args:
            epochs: Number of training epochs (default: 50)
            batch_size: Batch size for training (default: 16)
            imgsz: Image size for training (default: 640)
            patience: Early stopping patience in epochs (default: 10)
        """
        print("\n" + "="*70)
        print("🚀 YOLOV8 TRAINING START - NEU METAL SURFACE DEFECTS")
        print("="*70)
        print(f"Dataset YAML: {self.dataset_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {imgsz}x{imgsz}")
        print(f"Early Stopping Patience: {patience} epochs")
        print(f"Device: {self.device}")
        print("="*70 + "\n")
        
        try:
            # Train with full augmentation as per TRAINING_CONFIG.md
            results = self.model.train(
                data=self.dataset_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                patience=patience,  # Early stopping: stop if no improvement for N epochs
                device=self.device,
                project=self.project_name,
                name='run1',
                save=True,
                exist_ok=True,
                verbose=True,
                plots=True,
                # Augmentation parameters (matching TRAINING_CONFIG.md)
                hsv_h=0.015,  # HSV-Hue augmentation
                hsv_s=0.7,    # HSV-Saturation augmentation
                hsv_v=0.4,    # HSV-Value augmentation
                degrees=10,   # Rotation: +/- 10 degrees
                translate=0.1,  # Translation: 10%
                scale=0.5,    # Scale: 50%
                flipud=0.5,   # Flip upside down: 50%
                fliplr=0.5,   # Flip left-right: 50%
                mosaic=1.0,   # Mosaic augmentation: 100%
                # Training optimization
                optimizer='auto',
                lr0=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                # Validation and checkpointing
                val=True,
                save_period=1,
                cache=True,
            )
            
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"Results saved to: {self.project_name}/run1/")
            print("="*70 + "\n")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Training error: {str(e)}")
            print("Please ensure data.yaml path and dataset structure are correct.")
            return None
    
    def validate(self):
        """Validate trained model on validation set"""
        print("\n" + "="*70)
        print("🔍 VALIDATION START")
        print("="*70)
        
        try:
            metrics = self.model.val()
            print("\n" + "="*70)
            print("✅ VALIDATION COMPLETE")
            print("="*70)
            return metrics
        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
            return None
    
    def export_model(self, format='onnx'):
        """Export model to different formats for deployment"""
        print(f"\n📦 Exporting model to {format.upper()}")
        
        try:
            exported_path = self.model.export(format=format)
            print(f"✅ Model exported successfully to: {exported_path}")
            return exported_path
        except Exception as e:
            print(f"❌ Export error: {str(e)}")
            return None


def main():
    """Main training pipeline - matches TRAINING_CONFIG.md specifications"""
    
    # Configuration (matches TRAINING_CONFIG.md)
    DATA_YAML = "data.yaml"
    EPOCHS = 50
    BATCH_SIZE = 16
    IMGSZ = 640
    PATIENCE = 10  # Early stopping
    
    print("🎯 NEU YOLOV8 TRAINING PIPELINE")
    print("="*70)
    print(f"Configuration from TRAINING_CONFIG.md:")
    print(f"  Model: YOLOv8 Nano (lightweight, fast training)")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMGSZ}x{IMGSZ}")
    print(f"  Early Stopping Patience: {PATIENCE} epochs")
    print(f"  Augmentation: HSV, Rotation, Translation, Scale, Flip, Mosaic")
    print("="*70)
    
    # Verify data.yaml exists
    if not os.path.exists(DATA_YAML):
        print(f"\n❌ Error: {DATA_YAML} not found!")
        print("Please ensure data.yaml is in the current directory.")
        print(f"Expected path: {os.path.abspath(DATA_YAML)}")
        return
    
    print(f"✅ Found {DATA_YAML}")
    
    # Initialize trainer
    trainer = NEUYOLOTrainer(DATA_YAML)
    
    # Train model
    results = trainer.train(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        imgsz=IMGSZ,
        patience=PATIENCE
    )
    
    # If training successful, validate and export
    if results:
        print("\n📊 Post-Training Steps...")
        
        # Validate
        metrics = trainer.validate()
        
        # Export model for deployment
        trainer.export_model('onnx')
        
        print("\n✅ TRAINING PIPELINE COMPLETE!")
        print(f"Results saved in: ./{trainer.project_name}/run1/")
        print("Check results.png, confusion_matrix.png for performance visualization")
    else:
        print("\n❌ Training failed. Please check the error messages above.")


if __name__ == "__main__":
    main()