import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
import os
from datetime import datetime

# Import crowd monitoring functions
from utils.crowd_monitor import (
    detect_people, 
    log_crowd_detection, 
    get_crowd_detection_history, 
    get_crowd_stats,
    CROWD_THRESHOLDS
)

# Set page config
st.set_page_config(
    page_title="Crowd Monitoring - Rail Raksha",
    page_icon="👥",
    layout="wide"
)

# Create necessary directories if they don't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def main():
    st.title("Crowd Monitoring")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        model_path = st.text_input("Custom Model Path (optional)", "")
        
        # Display crowd level thresholds
        st.header("Crowd Level Thresholds")
        for level, threshold in CROWD_THRESHOLDS.items():
            if level != "Overcrowded":
                st.write(f"{level}: 0-{threshold} people")
            else:
                st.write(f"{level}: >{list(CROWD_THRESHOLDS.values())[-2]} people")
        
        # Input source selection
        st.header("Input Source")
        input_source = st.radio("Select Input Source", ["Upload Image", "Webcam"])
    
    # Main content
    if input_source == "Upload Image":
        st.write("Upload an image to analyze crowd density")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Save uploaded file
            file_path = UPLOAD_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display the uploaded file
            st.image(uploaded_file, caption="Original Image", use_column_width=True)
            
            # Process with crowd detection
            if st.button("Analyze Crowd"):
                with st.spinner("Analyzing crowd..."):
                    try:
                        # Run crowd detection
                        annotated_image, detections, people_count, crowd_level = detect_people(
                            str(file_path), 
                            model_path if model_path else None,
                            conf_threshold
                        )
                        
                        # Display the image with detections
                        st.image(annotated_image, caption="Crowd Analysis", use_column_width=True)
                        
                        # Display detection results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("People Count", people_count)
                        with col2:
                            st.metric("Crowd Level", crowd_level)
                        with col3:
                            st.metric("Detection Confidence", f"{np.mean([d['score'] for d in detections]):.2f}" if detections else "N/A")
                        
                        # Log the detection
                        log_crowd_detection(str(file_path), people_count, crowd_level)
                        st.success("Crowd detection logged successfully.")
                        
                    except Exception as e:
                        st.error(f"Error during crowd analysis: {str(e)}")
    else:  # Webcam input
        st.write("Use your webcam to analyze crowd density")
        
        # Webcam capture options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Capture Image"):
                with st.spinner("Capturing image..."):
                    # Import webcam utilities
                    from utils.webcam_utils import capture_webcam_frame, save_webcam_frame
                    
                    success, frame = capture_webcam_frame()
                    
                    if success:
                        # Save the frame
                        file_path = save_webcam_frame(frame)
                        
                        # Display the captured image
                        st.image(frame, caption="Captured Image", use_column_width=True)
                        
                        # Process with crowd detection
                        if st.button("Analyze Crowd in Captured Image"):
                            with st.spinner("Analyzing crowd..."):
                                try:
                                    # Run crowd detection
                                    annotated_image, detections, people_count, crowd_level = detect_people(
                                        file_path, 
                                        model_path if model_path else None,
                                        conf_threshold
                                    )
                                    
                                    # Display the image with detections
                                    st.image(annotated_image, caption="Crowd Analysis", use_column_width=True)
                                    
                                    # Display detection results
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("People Count", people_count)
                                    with col2:
                                        st.metric("Crowd Level", crowd_level)
                                    with col3:
                                        st.metric("Detection Confidence", f"{np.mean([d['score'] for d in detections]):.2f}" if detections else "N/A")
                                    
                                    # Log the detection
                                    log_crowd_detection(file_path, people_count, crowd_level)
                                    st.success("Crowd detection logged successfully.")
                                    
                                except Exception as e:
                                    st.error(f"Error during crowd analysis: {str(e)}")
                    else:
                        st.error("Failed to capture image from webcam. Please check your camera connection.")
        
        with col2:
            if st.button("Start Live Stream"):
                # Import webcam utilities
                from utils.webcam_utils import webcam_stream, process_webcam_feed
                
                frame_placeholder, stop_button = webcam_stream()
                
                # Define a function to process each frame with crowd detection
                def process_frame(frame):
                    try:
                        # Save the frame temporarily
                        temp_path = save_webcam_frame(frame, "temp")
                        
                        # Run crowd detection
                        annotated_image, _, _, _ = detect_people(
                            temp_path, 
                            model_path if model_path else None,
                            conf_threshold
                        )
                        
                        # Clean up the temporary file
                        os.remove(temp_path)
                        
                        return annotated_image
                    except Exception:
                        return frame
                
                # Process the webcam feed
                process_webcam_feed(frame_placeholder, stop_button, process_frame)
    
    # Display crowd detection history
    st.header("Crowd Detection History")
    
    # Get detection data
    df = get_crowd_detection_history()
    stats = get_crowd_stats()
    
    if not df.empty:
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", stats['total_detections'])
        with col2:
            st.metric("Average People Count", f"{stats['avg_people_count']:.1f}")
        with col3:
            st.metric("Most Common Level", max(stats['crowd_levels'].items(), key=lambda x: x[1])[0] if stats['crowd_levels'] else "N/A")
        
        # Display data table
        st.subheader("Detection Data")
        st.dataframe(df)
        
        # Display charts
        st.subheader("Crowd Analysis Charts")
        
        # Crowd level distribution
        if stats['crowd_levels']:
            fig1 = px.pie(
                values=list(stats['crowd_levels'].values()),
                names=list(stats['crowd_levels'].keys()),
                title="Crowd Level Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # People count over time
        if len(df) > 1:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            fig2 = px.line(
                df,
                x="timestamp",
                y="people_count",
                title="People Count Over Time",
                labels={"timestamp": "Time", "people_count": "Number of People"}
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No crowd detection data available. Run some detections first.")

if __name__ == "__main__":
    main() 