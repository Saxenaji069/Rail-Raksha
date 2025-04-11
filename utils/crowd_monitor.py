import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
import time
from datetime import datetime

# Define crowd level thresholds
CROWD_THRESHOLDS = {
    "Safe": 10,      # 0-10 people
    "Moderate": 30,  # 11-30 people
    "Overcrowded": float('inf')  # 31+ people
}

def detect_people(image_path, model_path=None, conf_threshold=0.25):
    """
    Detect people in an image using YOLO.
    
    Args:
        image_path (str): Path to the image file
        model_path (str, optional): Path to a custom YOLO model
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        tuple: (annotated_image, detections, people_count, crowd_level)
    """
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Convert to RGB for YOLO
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use YOLO to detect objects
    try:
        # Import here to avoid circular imports
        from models.yolo_model import detect_people as yolo_detect_people
        
        # Run detection
        annotated_image, detections, people_count = yolo_detect_people(
            str(image_path), 
            model_path,
            conf_threshold
        )
        
        # Determine crowd level
        crowd_level = classify_crowd(people_count)
        
        # Add crowd level label to the image
        annotated_image = add_crowd_label(annotated_image, people_count, crowd_level)
        
        return annotated_image, detections, people_count, crowd_level
        
    except Exception as e:
        print(f"Error during people detection: {str(e)}")
        return image_rgb, [], 0, "Unknown"

def classify_crowd(people_count):
    """
    Classify the crowd level based on the number of people.
    
    Args:
        people_count (int): Number of people detected
        
    Returns:
        str: Crowd level classification
    """
    for level, threshold in CROWD_THRESHOLDS.items():
        if people_count <= threshold:
            return level
    
    return "Overcrowded"  # Fallback

def add_crowd_label(image, people_count, crowd_level):
    """
    Add crowd information label to the image.
    
    Args:
        image (np.ndarray): The image to annotate
        people_count (int): Number of people detected
        crowd_level (str): Crowd level classification
        
    Returns:
        np.ndarray: Annotated image
    """
    # Create a copy of the image
    annotated = image.copy()
    
    # Define colors for different crowd levels
    colors = {
        "Safe": (0, 255, 0),      # Green
        "Moderate": (0, 165, 255), # Orange
        "Overcrowded": (0, 0, 255) # Red
    }
    
    # Get color for current crowd level
    color = colors.get(crowd_level, (255, 255, 255))
    
    # Add text to the image
    text = f"People: {people_count} | Crowd Level: {crowd_level}"
    cv2.putText(
        annotated, 
        text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        color, 
        2
    )
    
    return annotated

def log_crowd_detection(image_path, people_count, crowd_level):
    """
    Log crowd detection results to a CSV file.
    
    Args:
        image_path (str): Path to the image file
        people_count (int): Number of people detected
        crowd_level (str): Crowd level classification
        
    Returns:
        bool: True if logging was successful
    """
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Define the log file path
    log_file = data_dir / "crowd_detections.csv"
    
    # Create a new log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_filename = os.path.basename(image_path)
    
    log_entry = {
        "timestamp": timestamp,
        "image_filename": image_filename,
        "people_count": people_count,
        "crowd_level": crowd_level
    }
    
    # Convert to DataFrame
    df_entry = pd.DataFrame([log_entry])
    
    # Append to existing file or create new one
    if log_file.exists():
        df_entry.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_entry.to_csv(log_file, index=False)
    
    return True

def get_crowd_detection_history():
    """
    Get the history of crowd detections.
    
    Returns:
        pd.DataFrame: DataFrame containing crowd detection history
    """
    log_file = Path("data") / "crowd_detections.csv"
    
    if log_file.exists():
        return pd.read_csv(log_file, on_bad_lines='skip')
    else:
        return pd.DataFrame(columns=["timestamp", "image_filename", "people_count", "crowd_level"])

def get_crowd_stats():
    """
    Get statistics about crowd detections.
    
    Returns:
        dict: Dictionary containing crowd detection statistics
    """
    df = get_crowd_detection_history()
    
    if df.empty:
        return {
            "total_detections": 0,
            "avg_people_count": 0,
            "crowd_levels": {},
            "detection_dates": []
        }
    
    # Calculate statistics
    stats = {
        "total_detections": len(df),
        "avg_people_count": df["people_count"].mean(),
        "crowd_levels": df["crowd_level"].value_counts().to_dict(),
        "detection_dates": df["timestamp"].unique().tolist()
    }
    
    return stats 