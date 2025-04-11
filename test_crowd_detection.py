import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import os

# Import crowd monitoring functions
from utils.crowd_monitor import detect_people, log_crowd_detection

# Set page config
st.set_page_config(
    page_title="Crowd Detection Test",
    page_icon="👥",
    layout="wide"
)

def main():
    st.title("Crowd Detection Test")
    
    # Path to the sample image
    image_path = "uploads/crowd_sample.jpeg"
    
    # Check if the image exists
    if not os.path.exists(image_path):
        st.error(f"Image not found at {image_path}")
        return
    
    # Display the original image
    st.subheader("Original Image")
    st.image(image_path, caption="Original Image", use_container_width=True)
    
    # Process with crowd detection
    if st.button("Detect People"):
        with st.spinner("Analyzing crowd..."):
            try:
                # Run crowd detection
                annotated_image, detections, people_count, crowd_level = detect_people(
                    image_path, 
                    model_path=None,
                    conf_threshold=0.25
                )
                
                # Display the image with detections
                st.subheader("Detected People")
                st.image(annotated_image, caption="Crowd Analysis", use_container_width=True)
                
                # Display detection results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("People Count", people_count)
                with col2:
                    st.metric("Crowd Level", crowd_level)
                with col3:
                    st.metric("Detection Confidence", f"{np.mean([d['score'] for d in detections]):.2f}" if detections else "N/A")
                
                # Log the detection
                log_crowd_detection(image_path, people_count, crowd_level)
                st.success("Crowd detection logged successfully.")
                
                # Display individual detections
                if detections:
                    st.subheader("Individual Detections")
                    for i, det in enumerate(detections):
                        st.write(f"{i+1}. Person (Confidence: {det['score']:.2f})")
                
            except Exception as e:
                st.error(f"Error during crowd analysis: {str(e)}")

if __name__ == "__main__":
    main() 