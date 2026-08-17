"""
YOLOv8 Training Script for NEU Metal Surface Defects Detection
"""

from ultralytics import YOLO
import torch

class NEUYOLOTrainer:
    def __init__(self, dataset_yaml_path):
        self.dataset_yaml = dataset_yaml_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        self.model = YOLO('yolov8n.pt')
        
    def train(self, epochs=50, batch_size=16):
        results = self.model.train(
            data=self.dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            device=self.device,
            project='NEU_Detection',
            name='run1',
            save=True,
            plots=True
        )
        return results

if __name__ == "__main__":
    trainer = NEUYOLOTrainer('data.yaml')
    results = trainer.train()