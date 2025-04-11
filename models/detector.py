from ultralytics import YOLO
import numpy as np
from pathlib import Path

class RailDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the RailDetector with a YOLO model.
        
        Args:
            model_path (str, optional): Path to the YOLO model weights.
                                      If None, uses the default YOLOv8n model.
        """
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Use default YOLOv8n model
            self.model = YOLO('yolov8n.pt')
    
    def detect(self, image: np.ndarray) -> dict:
        """
        Perform object detection on the input image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            dict: Detection results containing boxes, scores, and class IDs
        """
        results = self.model(image)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                detections.append({
                    'box': box.tolist(),
                    'score': float(score),
                    'class_id': int(class_id),
                    'class_name': result.names[int(class_id)]
                })
        
        return {'detections': detections}
    
    def train(self, data_yaml: str, epochs: int = 100):
        """
        Train the model on custom data.
        
        Args:
            data_yaml (str): Path to data.yaml file
            epochs (int): Number of training epochs
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=16
        ) 