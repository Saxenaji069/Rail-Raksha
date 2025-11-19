import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import os
import time
from datetime import datetime, timedelta
import pandas as pd

# Constants for crowd monitoring
CROWD_THRESHOLDS = {
    'SAFE': 5,
    'MODERATE': 10,
    'OVERCROWDED': 10
}

YELLOW_LINE_Y = 400

@st.cache_resource
def load_detection_model():
    model = YOLO('yolov8n.pt')
    return model

def get_crowd_level(people_count):
    if people_count < CROWD_THRESHOLDS['SAFE']:
        return "Safe"
    elif people_count < CROWD_THRESHOLDS['MODERATE']:
        return "Moderate"
    else:
        return "Overcrowded"

def detect_people(frame, conf_threshold=0.25):
    """
    Detect people in a frame using YOLOv8.
    
    Args:
        frame (np.ndarray): The frame to process
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        tuple: (processed_frame, detections, people_count, crowd_level, violations)
    """
    # Ensure frame is in the right format
    if frame is None:
        print("ERROR: Frame is None")
        return None, [], 0, "Unknown", []
    
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model = load_detection_model()
    results = model(frame_rgb, conf=conf_threshold)
    
    # Debug prints
    print("DEBUG: YOLO Results Length:", len(results))
    for r in results:
        print("Boxes Found:", len(r.boxes))
    
    people_count = 0
    detections = []
    violations = []
    processed_frame = frame.copy()
    
    for result in results:
        boxes = result.boxes
        print(f"DEBUG: Processing {len(boxes)} boxes")
        
        for box in boxes:
            class_id = int(box.cls[0])
            name = result.names[class_id]
            conf = float(box.conf[0])
            
            print(f"DEBUG: Detected {name} with confidence {conf:.2f}")
            
            if name == 'person' and conf >= conf_threshold:
                people_count += 1
                print(f"✅ Person detected! Confidence: {conf:.2f}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'class': name,
                    'confidence': conf
                })
                
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} {conf:.2f}"
                cv2.putText(processed_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    print(f"DEBUG: Total people detected: {people_count}")
    crowd_level = get_crowd_level(people_count)
    add_to_detection_history(people_count, crowd_level, violations)
    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    return processed_frame_rgb, detections, people_count, crowd_level, violations

def check_yellow_line_violations(detections, yellow_line_y):
    violations = []
    already_counted = []

    for det in detections:
        try:
            box = det['box']
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if center_y > yellow_line_y:
                is_new_violation = True
                for vx, vy in already_counted:
                    if abs(center_x - vx) < 30 and abs(center_y - vy) < 30:
                        is_new_violation = False
                        break

                if is_new_violation:
                    violations.append((center_x, center_y))
                    already_counted.append((center_x, center_y))
        except Exception as e:
            print(f"Error processing violation: {str(e)}")
            continue

    return violations

def add_yellow_line(frame, y_coordinate):
    frame_with_line = frame.copy()
    height, width = frame.shape[:2]
    cv2.line(frame_with_line, (0, y_coordinate), (width, y_coordinate), (0, 255, 255), 2)
    return frame_with_line

def add_violation_markers(frame, violations):
    frame_with_markers = frame.copy()
    for x, y in violations:
        cv2.circle(frame_with_markers, (x, y), 5, (0, 0, 255), -1)
    return frame_with_markers

def log_crowd_detection(image_path, people_count, crowd_level, violations):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Detection Results:")
    print(f"- Image: {image_path}")
    print(f"- People Count: {people_count}")
    print(f"- Crowd Level: {crowd_level}")
    print(f"- Violations: {len(violations)}")
    add_to_detection_history(people_count, crowd_level, violations)

detection_history = []

def get_crowd_detection_history():
    if not detection_history:
        return pd.DataFrame(columns=['timestamp', 'people_count', 'crowd_level', 'violations'])

    df = pd.DataFrame(detection_history)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df

def add_to_detection_history(people_count, crowd_level, violations):
    timestamp = datetime.now()
    detection_record = {
        'timestamp': timestamp,
        'people_count': people_count,
        'crowd_level': crowd_level,
        'violations': violations
    }
    detection_history.append(detection_record)

    if len(detection_history) > 100:
        detection_history.pop(0)

def get_crowd_stats(time_window_minutes=60):
    """
    Get statistical analysis of crowd detection data within a time window.
    
    Args:
        time_window_minutes (int): Time window in minutes to analyze
        
    Returns:
        dict: Dictionary containing crowd statistics
    """
    if not detection_history:
        return {
            'total_detections': 0,
            'average_count': 0,
            'max_count': 0,
            'total_violations': 0,
            'crowd_level_distribution': {'Safe': 0, 'Moderate': 0, 'Overcrowded': 0},
            'peak_hours': [],
            'violation_trend': [],
            'avg_confidence': 0.0,
            'object_counts': {},
            'detection_dates': []
        }
    
    time_threshold = datetime.now() - timedelta(minutes=time_window_minutes)
    recent_records = [record for record in detection_history if record['timestamp'] >= time_threshold]
    
    if not recent_records:
        return {
            'total_detections': 0,
            'average_count': 0,
            'max_count': 0,
            'total_violations': 0,
            'crowd_level_distribution': {'Safe': 0, 'Moderate': 0, 'Overcrowded': 0},
            'peak_hours': [],
            'violation_trend': [],
            'avg_confidence': 0.0,
            'object_counts': {},
            'detection_dates': []
        }
    
    people_counts = [record['people_count'] for record in recent_records]
    average_count = sum(people_counts) / len(people_counts)
    max_count = max(people_counts)
    total_detections = len(recent_records)
    total_violations = sum(len(record['violations']) for record in recent_records)
    
    # Calculate average confidence
    confidences = []
    for record in recent_records:
        if 'detections' in record:
            for det in record['detections']:
                if 'confidence' in det:
                    confidences.append(det['confidence'])
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Get object counts
    object_counts = {'person': total_detections}
    
    # Get detection dates
    detection_dates = [record['timestamp'].strftime('%Y-%m-%d %H:%M:%S') for record in recent_records]
    
    crowd_levels = {'Safe': 0, 'Moderate': 0, 'Overcrowded': 0}
    for record in recent_records:
        crowd_levels[record['crowd_level']] += 1
    
    hourly_counts = {}
    for record in recent_records:
        hour = record['timestamp'].hour
        if hour not in hourly_counts:
            hourly_counts[hour] = []
        hourly_counts[hour].append(record['people_count'])
    
    peak_hours = []
    for hour, counts in hourly_counts.items():
        avg_count = sum(counts) / len(counts)
        if avg_count > average_count:
            peak_hours.append(hour)
    
    violation_trend = []
    for hour in range(24):
        hour_violations = sum(len(record['violations']) for record in recent_records if record['timestamp'].hour == hour)
        violation_trend.append(hour_violations)
    
    return {
        'total_detections': total_detections,
        'average_count': round(average_count, 2),
        'max_count': max_count,
        'total_violations': total_violations,
        'crowd_level_distribution': crowd_levels,
        'peak_hours': sorted(peak_hours),
        'violation_trend': violation_trend,
        'avg_confidence': round(avg_confidence, 2),
        'object_counts': object_counts,
        'detection_dates': detection_dates
    }
