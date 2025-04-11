import pandas as pd
import os
from datetime import datetime
from pathlib import Path

def log_detection(image_path, detections, log_file="data/detection_log.csv"):
    """
    Log detection results to a CSV file.
    
    Args:
        image_path (str): Path to the image file
        detections (list): List of detection dictionaries
        log_file (str): Path to the log file
    """
    # Create log file if it doesn't exist
    log_path = Path(log_file)
    if not log_path.exists():
        # Create directory if it doesn't exist
        log_path.parent.mkdir(exist_ok=True)
        
        # Create header
        with open(log_path, 'w') as f:
            f.write("timestamp,image_path,object_class,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2\n")
    
    # Prepare data for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_name = os.path.basename(image_path)
    
    # Create rows for each detection
    rows = []
    for det in detections:
        box = det['box']
        rows.append({
            'timestamp': timestamp,
            'image_path': image_name,
            'object_class': det['class_name'],
            'confidence': det['score'],
            'bbox_x1': box[0],
            'bbox_y1': box[1],
            'bbox_x2': box[2],
            'bbox_y2': box[3]
        })
    
    # Append to CSV
    df = pd.DataFrame(rows)
    df.to_csv(log_path, mode='a', header=False, index=False)
    
    return len(rows)

def get_detection_history(log_file="data/detection_log.csv"):
    """
    Get detection history from the log file.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        pd.DataFrame: Detection history
    """
    log_path = Path(log_file)
    if not log_path.exists():
        return pd.DataFrame()
    
    return pd.read_csv(log_path) 