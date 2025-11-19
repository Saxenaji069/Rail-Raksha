from ultralytics import YOLO
import numpy as np
from pathlib import Path

class RailDetector:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Initialize the RailDetector with a YOLO model.
        """
        model_path = Path(model_path)
        self.model = YOLO(str(model_path)) if model_path.exists() else YOLO('yolov8n.pt')
    
    def detect(self, image: np.ndarray) -> dict:
        """
        Perform object detection on the input image.
        """
        results = self.model(image)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, score, class_id in zip(boxes, scores, class_ids):
                detections.append({
                    'box': box.tolist(),           # [x1, y1, x2, y2]
                    'score': float(score),         # confidence
                    'class_id': int(class_id),
                    'class_name': result.names[int(class_id)]
                })

        return {'detections': detections}
    
    def train(self, data_yaml: str, epochs: int = 100):
        """
        Train the model on custom data.
        """
        self.model.train(data=data_yaml, epochs=epochs, imgsz=640, batch=16)
