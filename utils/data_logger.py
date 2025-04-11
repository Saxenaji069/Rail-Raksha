import pandas as pd
import os
from datetime import datetime
from pathlib import Path

# Define data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def log_detection(image_path, detections, log_file="data/detections.csv"):
    """
    Log detection results to a CSV file.
    
    Args:
        image_path (str): Path to the image file
        detections (list): List of detection dictionaries
        log_file (str): Path to the log file
    
    Returns:
        int: Number of detections logged
    """
    # Create log file if it doesn't exist
    log_path = Path(log_file)
    if not log_path.exists():
        # Create directory if it doesn't exist
        log_path.parent.mkdir(exist_ok=True)
        
        # Create header
        with open(log_path, 'w') as f:
            f.write("timestamp,image_filename,object_class,confidence\n")
    
    # Prepare data for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_filename = os.path.basename(image_path)
    
    # Create rows for each detection
    rows = []
    for det in detections:
        rows.append({
            'timestamp': timestamp,
            'image_filename': image_filename,
            'object_class': det['class_name'],
            'confidence': det['score']
        })
    
    # Append to CSV
    df = pd.DataFrame(rows)
    df.to_csv(log_path, mode='a', header=False, index=False)
    
    return len(rows)

def get_detection_history(log_file="data/detections.csv"):
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
    
    return pd.read_csv(log_path, on_bad_lines='skip')

def get_detection_stats(log_file="data/detections.csv"):
    """
    Get statistics about detections.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        dict: Statistics about detections
    """
    df = get_detection_history(log_file)
    
    if df.empty:
        return {
            'total_detections': 0,
            'object_counts': {},
            'avg_confidence': 0,
            'detection_dates': []
        }
    
    # Calculate statistics
    total_detections = len(df)
    object_counts = df['object_class'].value_counts().to_dict()
    avg_confidence = df['confidence'].mean()
    detection_dates = df['timestamp'].unique().tolist()
    
    return {
        'total_detections': total_detections,
        'object_counts': object_counts,
        'avg_confidence': avg_confidence,
        'detection_dates': detection_dates
    } 