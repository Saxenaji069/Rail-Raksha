import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import time
from collections import Counter
import colorsys
from ultralytics import YOLO

# Generate distinct colors for different classes
def generate_class_colors(num_classes):
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(x * 255) for x in rgb])
    return colors

# Load YOLOv5 model with caching
@st.cache_resource
def load_yolo_model():
    """Load YOLOv5 model from ultralytics with caching"""
    model = YOLO('yolov8n.pt')  # Load YOLOv8 nano model
    return model

def capture_webcam_frame():
    """
    Capture a frame from the webcam.
    
    Returns:
        tuple: (success, frame) where success is a boolean and frame is the captured image
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        return False, None
    
    # Read a frame
    ret, frame = cap.read()
    
    # Release the webcam
    cap.release()
    
    if not ret:
        return False, None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return True, frame_rgb

def save_webcam_frame(frame, prefix="webcam"):
    """
    Save a webcam frame to disk.
    
    Args:
        frame (np.ndarray): The frame to save
        prefix (str): Prefix for the filename
        
    Returns:
        str: Path to the saved image
    """
    import os
    from datetime import datetime
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"uploads/{prefix}_{timestamp}.jpg"
    
    # Save the image
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    return filename

def process_frame(frame, model, class_colors):
    """Process a single frame with YOLOv8 detection"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize frame for better performance while maintaining aspect ratio
    height, width = frame_rgb.shape[:2]
    max_size = 640
    scale = max_size / max(height, width)
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
    
    # Run inference
    results = model(frame_rgb, conf=0.4)  # Confidence threshold of 0.4
    
    # Create a copy of the frame for drawing
    annotated_frame = frame_rgb.copy()
    
    # Update detection counter
    detection_counts = Counter()
    
    # Debug information for person detection
    print(f"DEBUG: {len(results)} results from YOLO")
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        print(f"DEBUG: {len(boxes)} total boxes")
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class name and confidence
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = result.names[class_id]
            
            print(f"Detected: {name} with confidence {conf:.2f}")
            
            if name == 'person':
                print("✅ Person detected!")
            
            # Update counter
            detection_counts[name] += 1
            
            # Get color for this class
            color = class_colors[hash(name) % len(class_colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{name} {conf:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1 = max(y1, label_size[1])
            
            # Draw label background
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - baseline),
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add person count to the top of the frame
    person_count = detection_counts['person']
    count_text = f"People Count: {person_count}"
    
    # Calculate text size and position for top center
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(count_text, font, font_scale, thickness)[0]
    text_x = (annotated_frame.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 10
    
    # Draw semi-transparent background for better visibility
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, 
                 (text_x - 5, text_y - text_size[1] - 5),
                 (text_x + text_size[0] + 5, text_y + 5),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
    
    # Draw the count text
    cv2.putText(annotated_frame, count_text, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness)
    
    return annotated_frame, detection_counts

def webcam_stream():
    """
    Stream webcam feed to Streamlit.
    
    Returns:
        tuple: (frame_placeholder, stop_button) for controlling the stream
    """
    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()
    
    # Create a stop button with session state
    stop_button = st.session_state.get("stop_webcam", False) or st.button("Stop Stream", key="stop_webcam")
    
    return frame_placeholder, stop_button

def process_webcam_feed(frame_placeholder, stop_button, confidence_threshold=0.4):
    """
    Process webcam feed with YOLOv8 detection
    
    Args:
        frame_placeholder (streamlit.empty): Placeholder for the webcam feed
        stop_button (streamlit.button): Button to stop the webcam
        confidence_threshold (float, optional): Confidence threshold for YOLOv8 detection
        
    Returns:
        bool: True if the feed was stopped, False otherwise
    """
    try:
        # Load model
        model = load_yolo_model()
        
        # Generate colors for classes (YOLOv8 has 80 classes)
        class_colors = generate_class_colors(80)
        
        # Create legend placeholder
        legend_placeholder = st.empty()
        
        # Initialize webcam if not already created
        if 'cap' not in locals() or not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        # Initialize detection counter
        total_detections = Counter()
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            # Debug information
            print("DEBUG: Frame Shape", frame.shape)
            
            # Process frame
            annotated_frame, frame_detections = process_frame(frame, model, class_colors)
            
            # Debug information
            print("DEBUG: Detection counts", frame_detections)
            
            # Update total detections
            total_detections.update(frame_detections)
            
            # Display top 3 most detected classes
            top_3 = total_detections.most_common(3)
            if top_3:
                legend_text = "Most Detected Objects:\n"
                for obj, count in top_3:
                    color = class_colors[hash(obj) % len(class_colors)]
                    color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
                    legend_text += f'<div style="color: {color_hex}">• {obj}: {count}</div>'
                legend_placeholder.markdown(legend_text, unsafe_allow_html=True)
            
            # Display frame
            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
            
            # Add delay to prevent high CPU usage and counting overload
            time.sleep(0.05)
        
        # Release webcam
        cap.release()
        
    except Exception as e:
        st.error(f"Error in webcam processing: {str(e)}")
        if 'cap' in locals():
            cap.release()
    
    return True