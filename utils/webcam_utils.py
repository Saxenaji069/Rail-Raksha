import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import time

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

def save_webcam_frame(frame, save_dir="uploads"):
    """
    Save a webcam frame to disk.
    
    Args:
        frame (np.ndarray): The frame to save
        save_dir (str): Directory to save the frame
        
    Returns:
        str: Path to the saved image
    """
    # Create directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"webcam_{timestamp}.jpg"
    filepath = save_path / filename
    
    # Save the image
    cv2.imwrite(str(filepath), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    return str(filepath)

def webcam_stream():
    """
    Stream webcam feed to Streamlit.
    
    Returns:
        tuple: (frame_placeholder, stop_button) for controlling the stream
    """
    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()
    
    # Create a stop button
    stop_button = st.button("Stop Webcam")
    
    return frame_placeholder, stop_button

def process_webcam_feed(frame_placeholder, stop_button, process_func=None):
    """
    Process webcam feed and display it in Streamlit.
    
    Args:
        frame_placeholder (streamlit.empty): Placeholder for the webcam feed
        stop_button (streamlit.button): Button to stop the webcam
        process_func (callable, optional): Function to process each frame
        
    Returns:
        bool: True if the feed was stopped, False otherwise
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        return True
    
    try:
        while True:
            # Read a frame
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame if a processing function is provided
            if process_func:
                frame_rgb = process_func(frame_rgb)
            
            # Display the frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Check if stop button was clicked
            if stop_button:
                break
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.1)
    
    finally:
        # Release the webcam
        cap.release()
    
    return True 