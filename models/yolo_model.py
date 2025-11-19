from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path
import streamlit as st
import os

# Global variable to store the model instance
_model = None

def get_model(model_path=None):
    """
    Get or initialize the YOLO model.
    Uses a singleton pattern to load the model only once.
    
    Args:
        model_path (str, optional): Path to the YOLO model weights.
                                  If None, uses the default YOLOv8n model.
    
    Returns:
        YOLO: The YOLO model instance
    """
    global _model
    if _model is None:
        if model_path and Path(model_path).exists():
            _model = YOLO(model_path)
        else:
            # Use default YOLOv8n model
            _model = YOLO('yolov8n.pt')
    return _model

def detect_objects(image_path, model_path=None, conf_threshold=0.25):
    """
    Detect objects in an image using YOLO.
    
    Args:
        image_path (str): Path to the image file
        model_path (str, optional): Path to the YOLO model weights
        conf_threshold (float): Confidence threshold for detections
    
    Returns:
        tuple: (image, detections)
            - image: The original image with bounding boxes drawn
            - detections: List of detected objects with their properties
    """
    # Load the model (will only load once)
    model = get_model(model_path)
    
    # Load and preprocess the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Convert to RGB for YOLO
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(image_rgb, conf=conf_threshold)
    
    # Process results
    detections = []
    annotated_image = image_rgb.copy()
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            try:
                # Safely unpack detection data
                if len(box) >= 4:
                    x1, y1, x2, y2 = box.astype(int)
                else:
                    print("⚠️ Invalid box format:", box)
                    continue
                
                class_name = result.names[int(class_id)]
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name}: {score:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detections.append({
                    'box': box.tolist(),
                    'score': float(score),
                    'class_id': int(class_id),
                    'class_name': class_name
                })
            except Exception as e:
                print(f"⚠️ Error processing detection: {str(e)}")
                continue
    
    return annotated_image, detections

def display_detections(image_path, model_path=None, conf_threshold=0.25):
    """
    Display detections in Streamlit.
    
    Args:
        image_path (str): Path to the image file
        model_path (str, optional): Path to the YOLO model weights
        conf_threshold (float): Confidence threshold for detections
    
    Returns:
        None
    """
    try:
        annotated_image, detections = detect_objects(
            image_path, model_path, conf_threshold
        )
        
        # Display the image with detections
        st.image(annotated_image, caption="Detected Objects", use_container_width=True)
        
        # Display detection results
        if detections:
            st.subheader("Detection Results")
            for i, det in enumerate(detections):
                st.write(f"{i+1}. {det['class_name']} (Confidence: {det['score']:.2f})")
        else:
            st.info("No objects detected in the image.")
            
    except Exception as e:
        st.error(f"Error during object detection: {str(e)}")

def detect_people(image_path, model_path=None, conf_threshold=0.25):
    """
    Detect people in an image using YOLOv5/YOLOv8.
    
    Args:
        image_path (str or np.ndarray): Path to the image file or a NumPy array containing the image
        model_path (str, optional): Path to a custom YOLO model
        conf_threshold (float): Confidence threshold for detections
    
    Returns:
        tuple: (annotated_image, detections, people_count)
    """
    # Check if image_path is a NumPy array or a file path
    if isinstance(image_path, np.ndarray):
        # Use the array directly
        image = image_path
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image
    else:
        # Load the image from file
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        # Convert to RGB for YOLO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load the YOLO model
    if model_path and os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        # Use the default YOLOv8n model
        model = YOLO('yolov8n.pt')
    
    # Run inference
    results = model(image_rgb, conf=conf_threshold)
    
    # Process results
    detections = []
    people_count = 0
    
    # Create a copy of the image for annotation
    annotated_image = image_rgb.copy()
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # Only process 'person' class
            if class_name.lower() in ['person', 'people']:
                people_count += 1
                
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"Person {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add to detections list
                detections.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })
    
    return annotated_image, detections, people_count

def classify_crowd(people_count):
    """
    Classify the crowd level based on the number of people.
    
    Args:
        people_count (int): Number of people detected
        
    Returns:
        str: Crowd level classification
    """
    if people_count <= 10:
        return "Safe"
    elif people_count <= 30:
        return "Moderate"
    else:
        return "Overcrowded"

def get_crowd_color(crowd_level):
    """
    Get the color for a crowd level.
    
    Args:
        crowd_level (str): Crowd level classification
        
    Returns:
        str: Color code
    """
    colors = {
        "Safe": "green",
        "Moderate": "orange",
        "Overcrowded": "red"
    }
    return colors.get(crowd_level, "black")

def calculate_crowd_percentage(people_count):
    """
    Calculate the crowd percentage for visualization.
    
    Args:
        people_count (int): Number of people detected
        
    Returns:
        float: Crowd percentage (0-100)
    """
    return min((people_count / 50) * 100, 100) 