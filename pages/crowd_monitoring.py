import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from datetime import datetime
import time

# Import crowd monitoring functions
from utils.crowd_monitor import (
    detect_people, 
    log_crowd_detection, 
    get_crowd_detection_history, 
    get_crowd_stats,
    CROWD_THRESHOLDS,
    YELLOW_LINE_Y
)
from utils.video_processor import VideoProcessor, process_webcam_feed
from components.sidebar import render_sidebar

# Set page config
st.set_page_config(
    page_title="Crowd Monitoring - Rail Raksha",
    page_icon="👥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #0D47A1;
        --accent-color: #64B5F6;
        --text-color: #FFFFFF;
        --dark-bg: #1E1E1E;
        --darker-bg: #141414;
        --hover-color: #2196F3;
        --button-gradient: linear-gradient(45deg, #1E88E5, #64B5F6);
        --card-bg: rgba(255, 255, 255, 0.05);
        --sidebar-gradient: linear-gradient(135deg, rgba(30, 136, 229, 0.8), rgba(13, 71, 161, 0.9));
        --menu-hover: rgba(255, 255, 255, 0.1);
        --menu-active: rgba(255, 255, 255, 0.15);
    }

    /* Main content styling */
    .main {
        background: var(--dark-bg) !important;
        color: var(--text-color) !important;
    }

    /* Configuration section styling */
    .config-section {
        background: var(--card-bg) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }

    .config-section:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3) !important;
    }

    .section-header {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: var(--text-color) !important;
        margin-bottom: 1rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    /* Main header styling */
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: var(--text-color) !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        text-shadow: 0 0 10px rgba(100, 181, 246, 0.5) !important;
        background: linear-gradient(to right, #FFFFFF, #64B5F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .alert-banner {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
        text-align: center;
    }
    .alert-danger {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
    }
    .alert-warning {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ffcc80;
    }
    .violation-marker {
        color: #c62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Create necessary directories if they don't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def display_safety_alerts(crowd_level, violations):
    """Display safety alerts based on crowd level and violations."""
    if crowd_level == "Overcrowded":
        st.markdown(
            '<div class="alert-banner alert-danger">⚠️ OVERCROWDING ALERT: High crowd density detected!</div>',
            unsafe_allow_html=True
        )
    
    if violations:
        st.markdown(
            f'<div class="alert-banner alert-warning">⚠️ YELLOW LINE VIOLATION: {len(violations)} person(s) detected beyond safety line!</div>',
            unsafe_allow_html=True
        )

def main():
    # Get current page from sidebar
    current_page = render_sidebar()
    
    st.markdown('<h1 class="main-header">👥 Crowd Monitoring</h1>', unsafe_allow_html=True)
    
    # Configuration section in the main content area
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">⚙️ Configuration</h3>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    model_path = st.text_input("Custom Model Path (optional)", "")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display crowd level thresholds
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📊 Crowd Level Thresholds</h3>', unsafe_allow_html=True)
    for level, threshold in CROWD_THRESHOLDS.items():
        if level != "Overcrowded":
            st.write(f"{level}: 0-{threshold} people")
        else:
            st.write(f"{level}: >{list(CROWD_THRESHOLDS.values())[-2]} people")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input source selection
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📥 Input Source</h3>', unsafe_allow_html=True)
    input_source = st.radio("Select Input Source", ["Upload Image", "Upload Video", "Webcam"])
    st.markdown('</div>', unsafe_allow_html=True)
    
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
            st.image(uploaded_file, caption="Original Image", use_container_width=True)
            
            # Process with crowd detection
            if st.button("Analyze Crowd"):
                with st.spinner("Analyzing crowd..."):
                    try:
                        # Run crowd detection
                        frame = cv2.imread(str(file_path))
                        result = detect_people(frame, conf_threshold=conf_threshold)

                        # Safely unpack based on returned length
                        if isinstance(result, tuple):
                            if len(result) == 5:
                                annotated_image, detections, people_count, crowd_level, violations = result
                            elif len(result) == 3:
                                annotated_image, detections, people_count = result
                                crowd_level = get_crowd_level(people_count)
                                violations = []
                            else:
                                st.error("❌ Unexpected number of return values from detect_people()")
                                return
                        else:
                            st.error("❌ detect_people did not return a tuple")
                            return
                        
                        # Display the image with detections
                        st.image(annotated_image, caption="Crowd Analysis", use_container_width=True)
                        
                        # Display detection results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("People Count", people_count)
                        with col2:
                            st.metric("Crowd Level", crowd_level)
                        with col3:
                            st.metric("Detection Confidence", f"{np.mean([d['confidence'] for d in detections]):.2f}" if detections else "N/A")
                        
                        # Display safety alerts
                        display_safety_alerts(crowd_level, violations)
                        
                        # Log the detection
                        log_crowd_detection(str(file_path), people_count, crowd_level, violations)
                        st.success("Crowd detection logged successfully.")
                        
                    except Exception as e:
                        st.error(f"Error during crowd analysis: {str(e)}")
    
    elif input_source == "Upload Video":
        st.write("Upload a video to analyze crowd density")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Save uploaded file
            file_path = UPLOAD_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display video information
            st.video(uploaded_file)
            
            # Process with crowd detection
            if st.button("Analyze Video"):
                with st.spinner("Analyzing video..."):
                    try:
                        # Initialize video processor
                        processor = VideoProcessor(conf_threshold, model_path if model_path else None)
                        
                        # Create frame placeholder
                        frame_placeholder = st.empty()
                        
                        # Define display callback
                        def display_frame(frame):
                            frame_placeholder.image(frame, channels="BGR")
                        
                        # Process video
                        processor.process_video(str(file_path), display_callback=display_frame)
                        
                        st.success("Video analysis complete!")
                        
                    except Exception as e:
                        st.error(f"Error during video analysis: {str(e)}")
    
    else:  # Webcam input
        st.write("Use your webcam to analyze crowd density in real-time")
        
        # Webcam options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Webcam"):
                # Create frame placeholder
                frame_placeholder = st.empty()
                
                # Define display callback
                def display_frame(frame):
                    frame_placeholder.image(frame, channels="BGR")
                
                # Process webcam feed
                processor, stop_button = process_webcam_feed(
                    conf_threshold=conf_threshold,
                    model_path=model_path if model_path else None,
                    display_callback=display_frame
                )
                
                # Display stop button
                if stop_button:
                    processor.stop_processing()
                    st.info("Webcam processing stopped.")
        
        with col2:
            if st.button("Capture Single Frame"):
                # Initialize video capture
                cap = cv2.VideoCapture(0)
                
                if cap.isOpened():
                    # Capture frame
                    ret, frame = cap.read()
                    
                    if ret:
                        # Save frame
                        file_path = UPLOAD_DIR / f"webcam_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(str(file_path), frame)
                        
                        # Display frame
                        st.image(frame, caption="Captured Frame", use_container_width=True)
                        
                        # Process with crowd detection
                        if st.button("Analyze Captured Frame"):
                            with st.spinner("Analyzing frame..."):
                                try:
                                    # Run crowd detection
                                    frame = cv2.imread(str(file_path))
                                    result = detect_people(frame, conf_threshold=conf_threshold)

                                    # Safely unpack based on returned length
                                    if isinstance(result, tuple):
                                        if len(result) == 5:
                                            annotated_image, detections, people_count, crowd_level, violations = result
                                        elif len(result) == 3:
                                            annotated_image, detections, people_count = result
                                            crowd_level = get_crowd_level(people_count)
                                            violations = []
                                        else:
                                            st.error("❌ Unexpected number of return values from detect_people()")
                                            return
                                    else:
                                        st.error("❌ detect_people did not return a tuple")
                                        return
                                    
                                    # Display the image with detections
                                    st.image(annotated_image, caption="Crowd Analysis", use_container_width=True)
                                    
                                    # Display detection results
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("People Count", people_count)
                                    with col2:
                                        st.metric("Crowd Level", crowd_level)
                                    with col3:
                                        st.metric("Detection Confidence", f"{np.mean([d['confidence'] for d in detections]):.2f}" if detections else "N/A")
                                    
                                    # Display safety alerts
                                    display_safety_alerts(crowd_level, violations)
                                    
                                    # Log the detection
                                    log_crowd_detection(str(file_path), people_count, crowd_level, violations)
                                    st.success("Crowd detection logged successfully.")
                                    
                                except Exception as e:
                                    st.error(f"Error during crowd analysis: {str(e)}")
                    else:
                        st.error("Failed to capture frame from webcam.")
                else:
                    st.error("Failed to open webcam.")
                
                # Release resources
                cap.release()
    
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
            st.metric("Average People Count", f"{stats['average_count']:.1f}")
        with col3:
            st.metric("Most Common Level", max(stats['crowd_level_distribution'].items(), key=lambda x: x[1])[0] if stats['crowd_level_distribution'] else "N/A")
        
        # Display data table
        st.subheader("Detection Data")
        st.dataframe(df)
        
        # Display charts
        st.subheader("Crowd Analysis Charts")
        
        # Crowd level distribution
        if stats['crowd_level_distribution']:
            fig1 = px.pie(
                values=list(stats['crowd_level_distribution'].values()),
                names=list(stats['crowd_level_distribution'].keys()),
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