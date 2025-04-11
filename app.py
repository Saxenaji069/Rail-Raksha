import os
import subprocess
import pkg_resources

required = {
    'plotly',
    'streamlit',
    'opencv-python',
    'numpy',
    'pandas',
    'ultralytics',
    'Pillow'
}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    subprocess.check_call(["python", "-m", "pip", "install", *missing])

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
import plotly.express as px
from datetime import datetime
import time
import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Import custom modules
from models.yolo_model import detect_people, classify_crowd, get_crowd_color, calculate_crowd_percentage
from utils.data_logger import log_detection, get_detection_history, get_detection_stats
from utils.mock_detectors import detect_crime, detect_cleanliness
from utils.webcam_utils import capture_webcam_frame, save_webcam_frame, webcam_stream, process_webcam_feed
from utils.generate_test_images import generate_fake_crowd_image, create_background_image

# Set page config
st.set_page_config(
    page_title="Rail Raksha",
    page_icon="🚂",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 500;
        color: #1565C0;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .info-text {
        font-size: 1.1rem;
        color: #424242;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# Create necessary directories if they don't exist
UPLOAD_DIR = Path("uploads")
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
UTILS_DIR = Path("utils")
PAGES_DIR = Path("pages")

for dir_path in [UPLOAD_DIR, MODEL_DIR, DATA_DIR, UTILS_DIR, PAGES_DIR]:
    dir_path.mkdir(exist_ok=True)

def home_page():
    """Display the home page content"""
    st.markdown('<h1 class="main-header">🚂 Rail Raksha - Railway Infrastructure Monitoring</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.write("""
    ## 👋 Welcome to Rail Raksha
    
    Rail Raksha is a comprehensive railway infrastructure monitoring system that uses 
    computer vision and deep learning to detect objects, assess cleanliness, and identify 
    potential security issues.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">🔑 Key Features:</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("""
        - **🔍 Object Detection**: Identify trains, people, vehicles, and infrastructure elements
        - **🚨 Crime Detection**: Monitor for suspicious activities and security threats
        - **🧹 Cleanliness Assessment**: Evaluate the condition of railway infrastructure
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("""
        - **👥 Crowd Monitoring**: Analyze crowd density and classify crowd levels
        - **📊 Data Logging**: Track and analyze detection results over time
        - **📸 Webcam Integration**: Real-time monitoring and analysis
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">📋 How to Use:</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.write("""
    1. Navigate to the "Upload & Detect" page to analyze images or use your webcam
    2. View detection results and analysis
    3. Check the "Detection Logs" page to see historical data
    4. Use the "Crowd Monitoring" page to analyze crowd density
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display some statistics
    stats = get_detection_stats()
    
    if stats['total_detections'] > 0:
        st.markdown('<h2 class="section-header">📊 System Statistics</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-value">', unsafe_allow_html=True)
            st.metric("Total Detections", stats['total_detections'])
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-value">', unsafe_allow_html=True)
            st.metric("Average Confidence", f"{stats['avg_confidence']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-value">', unsafe_allow_html=True)
            st.metric("Unique Images", len(stats['detection_dates']))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display a sample image if available
        if os.path.exists(UPLOAD_DIR) and any(os.path.isfile(os.path.join(UPLOAD_DIR, f)) for f in os.listdir(UPLOAD_DIR)):
            sample_images = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if sample_images:
                st.markdown('<h2 class="subsection-header">🖼️ Sample Image</h2>', unsafe_allow_html=True)
                sample_image = os.path.join(UPLOAD_DIR, sample_images[0])
                st.image(sample_image, caption="Sample Image", use_container_width=True)
    else:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.info("No detection data available yet. Run some detections to see statistics here.")
        st.markdown('</div>', unsafe_allow_html=True)

def upload_detect_page():
    """Display the upload and detect page content"""
    st.markdown('<h1 class="main-header">📤 Upload & Detect</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        model_path = st.text_input("Custom Model Path (optional)", "")
        
        st.markdown('<h2 class="sidebar-header">📥 Input Source</h2>', unsafe_allow_html=True)
        input_source = st.radio("Select Input Source", ["Upload Image", "Webcam", "Generate Test Image"])
        
        if input_source == "Generate Test Image":
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            num_people = st.slider("Number of People", 1, 50, 15)
            if st.button("Generate Random Crowd Image"):
                with st.spinner("Generating test image..."):
                    # Create background if it doesn't exist
                    if not os.path.exists(os.path.join(UPLOAD_DIR, "background.jpg")):
                        create_background_image()
                    
                    # Generate the test image
                    image_path = generate_fake_crowd_image(num_people)
                    st.session_state['generated_image_path'] = image_path
                    st.success(f"Generated test image with {num_people} people!")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    if input_source == "Upload Image":
        st.markdown('<h2 class="section-header">📤 Upload Images for Analysis</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png', 'mp4'])
        
        if uploaded_file is not None:
            # Save uploaded file
            file_path = UPLOAD_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display the uploaded file
            if uploaded_file.type.startswith('image'):
                # Display original image
                st.markdown('<h2 class="subsection-header">🖼️ Original Image</h2>', unsafe_allow_html=True)
                st.image(uploaded_file, caption="Original Image", use_container_width=True)
                
                # Process with YOLO
                if st.button("🔍 Detect Objects"):
                    with st.spinner("Detecting objects..."):
                        try:
                            # Use the YOLO model function
                            annotated_image, detections, people_count = detect_people(
                                str(file_path), 
                                model_path if model_path else None,
                                conf_threshold
                            )
                            
                            # Classify crowd level
                            crowd_level = classify_crowd(people_count)
                            crowd_color = get_crowd_color(crowd_level)
                            
                            # Display the image with detections
                            st.markdown('<h2 class="subsection-header">🔍 Detection Results</h2>', unsafe_allow_html=True)
                            st.image(annotated_image, caption="Detected Objects", use_container_width=True)
                            
                            # Display detection results
                            if detections:
                                st.markdown('<h3 class="subsection-header">📊 Detection Details</h3>', unsafe_allow_html=True)
                                for i, det in enumerate(detections):
                                    st.write(f"{i+1}. {det['class_name']} (Confidence: {det['score']:.2f})")
                                
                                # Log detections
                                num_logged = log_detection(str(file_path), detections)
                                st.success(f"✅ Logged {num_logged} detections to history.")
                            else:
                                st.info("ℹ️ No objects detected in the image.")
                            
                            # Display people count and crowd level
                            st.markdown('<h3 class="subsection-header">👥 Crowd Analysis</h3>', unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown('<div class="metric-value">', unsafe_allow_html=True)
                                st.metric("People Count", people_count)
                                st.markdown('</div>', unsafe_allow_html=True)
                            with col2:
                                st.markdown(f'<h2 style="color:{crowd_color};">{crowd_level}</h2>', unsafe_allow_html=True)
                            
                            # Display crowd density meter
                            crowd_percent = calculate_crowd_percentage(people_count)
                            
                            # Create a custom progress bar with color
                            st.markdown('<h3 class="subsection-header">📊 Crowd Density Meter</h3>', unsafe_allow_html=True)
                            if crowd_percent <= 33:
                                color = "green"
                            elif crowd_percent <= 66:
                                color = "orange"
                            else:
                                color = "red"
                                
                            st.markdown(
                                f"""
                                <div style="background-color: #f0f0f0; height: 30px; border-radius: 5px; margin-bottom: 10px;">
                                    <div style="background-color: {color}; width: {crowd_percent}%; height: 100%; border-radius: 5px;"></div>
                                </div>
                                <div style="text-align: center; font-size: 1.2rem; font-weight: bold;">{crowd_percent:.1f}%</div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Run additional analyses
                            st.markdown("---")
                            st.markdown('<h3 class="subsection-header">🔍 Additional Analysis</h3>', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                                st.subheader("🚨 Crime Detection")
                                crime_result = detect_crime(str(file_path))
                                if crime_result["status"] == "no crime":
                                    st.success(f"✅ No crime detected (Confidence: {crime_result['confidence']:.2f})")
                                else:
                                    st.error(f"⚠️ Crime detected: {crime_result['status']} (Confidence: {crime_result['confidence']:.2f})")
                                    st.write(f"📍 Location: {crime_result['location']}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                                st.subheader("🧹 Cleanliness Assessment")
                                cleanliness_result = detect_cleanliness(str(file_path))
                                st.write(f"📊 Score: {cleanliness_result['score']}/100")
                                st.write(f"📋 Level: {cleanliness_result['level']}")
                                
                                if cleanliness_result['issues']:
                                    st.write("⚠️ Issues detected:")
                                    for issue in cleanliness_result['issues']:
                                        st.write(f"- {issue}")
                                else:
                                    st.write("✅ No issues detected.")
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.error(f"❌ Error during object detection: {str(e)}")
            elif uploaded_file.type.startswith('video'):
                st.video(uploaded_file)
                st.info("ℹ️ Video processing is not yet implemented.")
    elif input_source == "Webcam":
        st.markdown('<h2 class="section-header">📸 Webcam Capture</h2>', unsafe_allow_html=True)
        
        # Webcam capture options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📸 Capture Image"):
                with st.spinner("Capturing image..."):
                    success, frame = capture_webcam_frame()
                    
                    if success:
                        # Save the frame
                        file_path = save_webcam_frame(frame)
                        
                        # Display the captured image
                        st.markdown('<h2 class="subsection-header">🖼️ Captured Image</h2>', unsafe_allow_html=True)
                        st.image(frame, caption="Captured Image", use_container_width=True)
                        
                        # Process with YOLO
                        if st.button("🔍 Detect Objects in Captured Image"):
                            with st.spinner("Detecting objects..."):
                                try:
                                    # Use the YOLO model function
                                    annotated_image, detections, people_count = detect_people(
                                        file_path, 
                                        model_path if model_path else None,
                                        conf_threshold
                                    )
                                    
                                    # Classify crowd level
                                    crowd_level = classify_crowd(people_count)
                                    crowd_color = get_crowd_color(crowd_level)
                                    
                                    # Display the image with detections
                                    st.markdown('<h2 class="subsection-header">🔍 Detection Results</h2>', unsafe_allow_html=True)
                                    st.image(annotated_image, caption="Detected Objects", use_container_width=True)
                                    
                                    # Display detection results
                                    if detections:
                                        st.markdown('<h3 class="subsection-header">📊 Detection Details</h3>', unsafe_allow_html=True)
                                        for i, det in enumerate(detections):
                                            st.write(f"{i+1}. {det['class_name']} (Confidence: {det['score']:.2f})")
                                        
                                        # Log detections
                                        num_logged = log_detection(file_path, detections)
                                        st.success(f"✅ Logged {num_logged} detections to history.")
                                    else:
                                        st.info("ℹ️ No objects detected in the image.")
                                    
                                    # Display people count and crowd level
                                    st.markdown('<h3 class="subsection-header">👥 Crowd Analysis</h3>', unsafe_allow_html=True)
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown('<div class="metric-value">', unsafe_allow_html=True)
                                        st.metric("People Count", people_count)
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    with col2:
                                        st.markdown(f'<h2 style="color:{crowd_color};">{crowd_level}</h2>', unsafe_allow_html=True)
                                    
                                    # Display crowd density meter
                                    crowd_percent = calculate_crowd_percentage(people_count)
                                    
                                    # Create a custom progress bar with color
                                    st.markdown('<h3 class="subsection-header">📊 Crowd Density Meter</h3>', unsafe_allow_html=True)
                                    if crowd_percent <= 33:
                                        color = "green"
                                    elif crowd_percent <= 66:
                                        color = "orange"
                                    else:
                                        color = "red"
                                        
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f0f0f0; height: 30px; border-radius: 5px; margin-bottom: 10px;">
                                            <div style="background-color: {color}; width: {crowd_percent}%; height: 100%; border-radius: 5px;"></div>
                                        </div>
                                        <div style="text-align: center; font-size: 1.2rem; font-weight: bold;">{crowd_percent:.1f}%</div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Run additional analyses
                                    st.markdown("---")
                                    st.markdown('<h3 class="subsection-header">🔍 Additional Analysis</h3>', unsafe_allow_html=True)
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown('<div class="highlight">', unsafe_allow_html=True)
                                        st.subheader("🚨 Crime Detection")
                                        crime_result = detect_crime(file_path)
                                        if crime_result["status"] == "no crime":
                                            st.success(f"✅ No crime detected (Confidence: {crime_result['confidence']:.2f})")
                                        else:
                                            st.error(f"⚠️ Crime detected: {crime_result['status']} (Confidence: {crime_result['confidence']:.2f})")
                                            st.write(f"📍 Location: {crime_result['location']}")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    with col2:
                                        st.markdown('<div class="highlight">', unsafe_allow_html=True)
                                        st.subheader("🧹 Cleanliness Assessment")
                                        cleanliness_result = detect_cleanliness(file_path)
                                        st.write(f"📊 Score: {cleanliness_result['score']}/100")
                                        st.write(f"📋 Level: {cleanliness_result['level']}")
                                        
                                        if cleanliness_result['issues']:
                                            st.write("⚠️ Issues detected:")
                                            for issue in cleanliness_result['issues']:
                                                st.write(f"- {issue}")
                                        else:
                                            st.write("✅ No issues detected.")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                        
                                except Exception as e:
                                    st.error(f"❌ Error during object detection: {str(e)}")
                    else:
                        st.error("❌ Failed to capture image from webcam. Please check your camera connection.")
        
        with col2:
            if st.button("🎥 Start Live Stream"):
                frame_placeholder, stop_button = webcam_stream()
                
                # Define a function to process each frame with YOLO
                def process_frame(frame):
                    try:
                        # Save the frame temporarily
                        temp_path = save_webcam_frame(frame, "temp")
                        
                        # Run YOLO detection
                        annotated_image, _, _ = detect_people(
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
    else:  # Generate Test Image
        if 'generated_image_path' in st.session_state:
            image_path = st.session_state['generated_image_path']
            
            # Display the generated image
            st.markdown('<h2 class="subsection-header">🖼️ Generated Test Image</h2>', unsafe_allow_html=True)
            st.image(image_path, caption="Generated Test Image", use_container_width=True)
            
            # Process with YOLO
            if st.button("🔍 Detect Objects in Generated Image"):
                with st.spinner("Detecting objects..."):
                    try:
                        # Use the YOLO model function
                        annotated_image, detections, people_count = detect_people(
                            image_path, 
                            model_path if model_path else None,
                            conf_threshold
                        )
                        
                        # Classify crowd level
                        crowd_level = classify_crowd(people_count)
                        crowd_color = get_crowd_color(crowd_level)
                        
                        # Display the image with detections
                        st.markdown('<h2 class="subsection-header">🔍 Detection Results</h2>', unsafe_allow_html=True)
                        st.image(annotated_image, caption="Detected Objects", use_container_width=True)
                        
                        # Display detection results
                        if detections:
                            st.markdown('<h3 class="subsection-header">📊 Detection Details</h3>', unsafe_allow_html=True)
                            for i, det in enumerate(detections):
                                st.write(f"{i+1}. {det['class_name']} (Confidence: {det['score']:.2f})")
                            
                            # Log detections
                            num_logged = log_detection(image_path, detections)
                            st.success(f"✅ Logged {num_logged} detections to history.")
                        else:
                            st.info("ℹ️ No objects detected in the image.")
                        
                        # Display people count and crowd level
                        st.markdown('<h3 class="subsection-header">👥 Crowd Analysis</h3>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="metric-value">', unsafe_allow_html=True)
                            st.metric("People Count", people_count)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<h2 style="color:{crowd_color};">{crowd_level}</h2>', unsafe_allow_html=True)
                        
                        # Display crowd density meter
                        crowd_percent = calculate_crowd_percentage(people_count)
                        
                        # Create a custom progress bar with color
                        st.markdown('<h3 class="subsection-header">📊 Crowd Density Meter</h3>', unsafe_allow_html=True)
                        if crowd_percent <= 33:
                            color = "green"
                        elif crowd_percent <= 66:
                            color = "orange"
                        else:
                            color = "red"
                            
                        st.markdown(
                            f"""
                            <div style="background-color: #f0f0f0; height: 30px; border-radius: 5px; margin-bottom: 10px;">
                                <div style="background-color: {color}; width: {crowd_percent}%; height: 100%; border-radius: 5px;"></div>
                            </div>
                            <div style="text-align: center; font-size: 1.2rem; font-weight: bold;">{crowd_percent:.1f}%</div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during object detection: {str(e)}")
        else:
            st.info("ℹ️ Generate a test image using the sidebar options.")

def detection_logs_page():
    """Display the detection logs page content"""
    st.markdown('<h1 class="main-header">📊 Detection Logs</h1>', unsafe_allow_html=True)
    
    # Get detection data
    df = get_detection_history()
    stats = get_detection_stats()
    
    if not df.empty:
        # Display summary statistics
        st.markdown('<h2 class="section-header">📈 Summary Statistics</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-value">', unsafe_allow_html=True)
            st.metric("Total Detections", stats['total_detections'])
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-value">', unsafe_allow_html=True)
            st.metric("Average Confidence", f"{stats['avg_confidence']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-value">', unsafe_allow_html=True)
            st.metric("Unique Images", len(df['image_filename'].unique()))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display data table
        st.markdown('<h2 class="section-header">📋 Detection Data</h2>', unsafe_allow_html=True)
        st.dataframe(df)
        
        # Display charts
        st.markdown('<h2 class="section-header">📊 Detection Charts</h2>', unsafe_allow_html=True)
        
        # Object class distribution
        if len(stats['object_counts']) > 0:
            st.markdown('<h3 class="subsection-header">🔍 Object Class Distribution</h3>', unsafe_allow_html=True)
            fig1 = px.bar(
                x=list(stats['object_counts'].keys()),
                y=list(stats['object_counts'].values()),
                title="Object Class Distribution",
                labels={"x": "Object Class", "y": "Count"}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # Confidence distribution
        st.markdown('<h3 class="subsection-header">📈 Confidence Distribution</h3>', unsafe_allow_html=True)
        fig2 = px.histogram(
            df, 
            x="confidence",
            title="Confidence Distribution",
            labels={"confidence": "Confidence", "count": "Count"}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Detections over time
        st.markdown('<h3 class="subsection-header">⏱️ Detections Over Time</h3>', unsafe_allow_html=True)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_counts = df.groupby('date').size().reset_index(name='count')
        fig3 = px.line(
            daily_counts,
            x="date",
            y="count",
            title="Detections Over Time",
            labels={"date": "Date", "count": "Number of Detections"}
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.info("ℹ️ No detection data available. Run some detections first.")
        st.markdown('</div>', unsafe_allow_html=True)

def about_page():
    """Display the about page content"""
    st.markdown('<h1 class="main-header">ℹ️ About Rail Raksha</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.write("""
    Rail Raksha is a web-based application for monitoring railway infrastructure using computer vision and deep learning.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">🔑 Features</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    - **📸 Image and video upload support**
    - **🔍 Real-time object detection using YOLO**
    - **🚨 Infrastructure defect detection**
    - **🚨 Crime and suspicious behavior detection**
    - **🧹 Cleanliness assessment**
    - **👥 Crowd monitoring and density analysis**
    - **📊 Detection history logging and analysis**
    - **📸 Webcam integration for live monitoring**
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">📋 How to Use</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.write("""
    1. Navigate to the "Upload & Detect" page
    2. Upload an image or use your webcam
    3. Click "Detect Objects" to run the detection
    4. View the results and detection history
    5. Use the "Crowd Monitoring" page to analyze crowd density
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">🛠️ Technologies Used</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    - **Streamlit** for the web interface
    - **YOLO** for object detection
    - **OpenCV** for image processing
    - **Pandas** and **Plotly** for data analysis and visualization
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Home", "Upload & Detect", "Detection Logs", "About"]
    )
    
    # Display the selected page
    if page == "Home":
        home_page()
    elif page == "Upload & Detect":
        upload_detect_page()
    elif page == "Detection Logs":
        detection_logs_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main() 
