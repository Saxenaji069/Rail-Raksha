import os
import subprocess
import pkg_resources
from collections import Counter
import colorsys

# Set environment variable to prevent file watcher conflicts with PyTorch
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

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

# Import custom modules
from models.yolo_model import detect_people, classify_crowd, get_crowd_color, calculate_crowd_percentage
from utils.data_logger import log_detection, get_detection_history, get_detection_stats
from utils.mock_detectors import detect_crime, detect_cleanliness
from utils.webcam_utils import capture_webcam_frame, save_webcam_frame, webcam_stream, process_webcam_feed
from utils.generate_test_images import generate_fake_crowd_image, create_background_image
from components.sidebar import render_sidebar
from utils.video_processor import VideoProcessor

# Set page config
st.set_page_config(
    page_title="Rail Raksha",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #1d4ed8;
        --accent-color: #60a5fa;
        --text-color: #f8fafc;
        --dark-bg: #0f172a;
        --darker-bg: #020617;
        --hover-color: #3b82f6;
        --button-gradient: linear-gradient(135deg, #2563eb, #60a5fa);
        --card-bg: rgba(255, 255, 255, 0.03);
        --sidebar-gradient: linear-gradient(135deg, rgba(37, 99, 235, 0.9), rgba(29, 78, 216, 0.95));
        --menu-hover: rgba(255, 255, 255, 0.08);
        --menu-active: rgba(255, 255, 255, 0.12);
        --success-color: #22c55e;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }

    /* Main content styling */
    .main {
        background: var(--dark-bg) !important;
        color: var(--text-color) !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    /* Configuration section styling */
    .config-section {
        background: var(--card-bg) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        margin: 1.5rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        backdrop-filter: blur(12px) !important;
    }

    .config-section:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    .section-header {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: var(--text-color) !important;
        margin-bottom: 1.5rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
        letter-spacing: -0.025em !important;
    }

    /* Main header styling */
    .main-header {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: var(--text-color) !important;
        text-align: center !important;
        margin-bottom: 2.5rem !important;
        text-shadow: 0 0 20px rgba(96, 165, 250, 0.5) !important;
        background: linear-gradient(135deg, #f8fafc, #60a5fa) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        letter-spacing: -0.05em !important;
    }

    /* Button styling */
    .stButton > button {
        background: var(--button-gradient) !important;
        color: var(--text-color) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: var(--text-color) !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2) !important;
    }

    /* Metric styling */
    .metric-value {
        background: var(--card-bg) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease !important;
    }

    .metric-value:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Info text styling */
    .info-text {
        color: var(--text-color) !important;
        font-size: 1.1rem !important;
        line-height: 1.7 !important;
        opacity: 0.9 !important;
    }

    /* Highlight box styling */
    .highlight {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin: 1rem 0 !important;
    }

    /* Status colors */
    .success-text { color: var(--success-color) !important; }
    .warning-text { color: var(--warning-color) !important; }
    .error-text { color: var(--error-color) !important; }

    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem !important;
        }
        
        .config-section {
            padding: 1.5rem !important;
        }
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
        st.info("ℹ️ No detection data available yet. Run some detections to see statistics here.")
        st.markdown('</div>', unsafe_allow_html=True)

def upload_detect_page():
    """Display the upload and detect page content"""
    st.markdown('<h1 class="main-header">📤 Upload & Detect</h1>', unsafe_allow_html=True)
    
    # Configuration section in the main content area instead of sidebar
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">⚙️ Model Settings</h3>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    model_path = st.text_input("Custom Model Path", "")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📥 Input Source</h3>', unsafe_allow_html=True)
    input_source = st.radio("", ["Upload Image", "Webcam", "Generate Test Image"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if input_source == "Generate Test Image":
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">👥 Test Image Settings</h3>', unsafe_allow_html=True)
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
            # Save the uploaded file
            file_path = UPLOAD_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Check if it's a video file
            is_video = uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov'))
            
            if is_video:
                # Video processing controls
                col1, col2 = st.columns(2)
                with col1:
                    process_button = st.button("Process Video")
                with col2:
                    stop_button = st.button("Stop Processing")
                
                # Create video processor instance
                processor = VideoProcessor(conf_threshold=conf_threshold)
                
                # Create placeholder for video display
                video_placeholder = st.empty()
                stats_placeholder = st.empty()
                
                if process_button:
                    try:
                        # Process video and display results
                        processor.process_video(
                            str(file_path),
                            display_callback=lambda frame: video_placeholder.image(frame, channels="BGR")
                        )
                        
                        # Display statistics
                        stats = processor.get_detection_stats()
                        with stats_placeholder.container():
                            st.subheader("Detection Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Frames", stats['total_frames'])
                                st.metric("Average People", f"{stats['avg_people']:.1f}")
                            with col2:
                                st.metric("Max People", stats['max_people'])
                                st.metric("Total People", processor.total_people)
                            with col3:
                                st.metric("Overcrowded Frames", stats['crowd_levels']['Overcrowded'])
                                st.metric("Moderate Frames", stats['crowd_levels']['Moderate'])
                            
                            # Display crowd level distribution
                            st.subheader("Crowd Level Distribution")
                            crowd_data = pd.DataFrame({
                                'Level': list(stats['crowd_levels'].keys()),
                                'Frames': list(stats['crowd_levels'].values())
                            })
                            st.bar_chart(crowd_data.set_index('Level'))
                    
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                
                if stop_button:
                    processor.is_processing = False
                    st.info("Video processing stopped.")
            
            else:
                # Image processing
                try:
                    # Process the image
                    frame = cv2.imread(str(file_path))
                    result = detect_people(frame)
                    
                    # Safely unpack based on returned length
                    if isinstance(result, tuple):
                        if len(result) == 5:
                            processed_frame, detections, people_count, crowd_level, violations = result
                        elif len(result) == 3:
                            processed_frame, detections, people_count = result
                            crowd_level = "Unknown"
                            violations = []
                        else:
                            st.error("❌ Unexpected number of return values from detect_people()")
                            return
                    else:
                        st.error("❌ detect_people did not return a tuple")
                        return
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(processed_frame, caption="Processed Image", use_container_width=True)
                    with col2:
                        st.subheader("Detection Results")
                        st.metric("People Count", people_count)
                        st.metric("Crowd Level", crowd_level)
                        if violations:
                            st.warning(f"Yellow Line Violations: {len(violations)}")
                        
                        # Display detection details
                        st.subheader("Detection Details")
                        for i, det in enumerate(detections):
                            st.write(f"Person {i+1}: Confidence {det['confidence']:.2f}")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    elif input_source == "Webcam":
        st.markdown('<h2 class="section-header">🎥 Live Object Detection</h2>', unsafe_allow_html=True)
        
        # Add description
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            This feature uses YOLOv5 to detect multiple object classes in real-time, including:
            - People
            - Vehicles
            - Bags/luggage
            - Infrastructure elements
            - And many more...
        </div>
        """, unsafe_allow_html=True)
        
        # Add confidence threshold slider
        conf_threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📸 Capture Single Frame"):
                with st.spinner("Capturing and processing frame..."):
                    success, frame = capture_webcam_frame()
                    
                    if success:
                        # Save the frame
                        file_path = save_webcam_frame(frame)
                        
                        # Display the captured frame
                        st.markdown('<h3>Captured Frame</h3>', unsafe_allow_html=True)
                        st.image(frame, caption="Captured Frame", use_container_width=True)
                    else:
                        st.error("Failed to capture frame from webcam")
        
        with col2:
            if st.button("🎥 Start Live Detection"):
                st.markdown('<h3>Live Detection Stream</h3>', unsafe_allow_html=True)
                frame_placeholder, stop_button = webcam_stream()
                
                # Start processing with our improved detection
                process_webcam_feed(frame_placeholder, stop_button, conf_threshold)
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
                        frame = cv2.imread(image_path)
                        processed_frame, detections, people_count, crowd_level, violations = detect_people(frame)
                        
                        # Display the image with detections
                        st.markdown('<h2 class="subsection-header">🔍 Detection Results</h2>', unsafe_allow_html=True)
                        st.image(processed_frame, caption="Detected Objects", use_container_width=True)
                        
                        # Display detection results
                        if detections:
                            st.markdown('<h3 class="subsection-header">📊 Detection Details</h3>', unsafe_allow_html=True)
                            for i, det in enumerate(detections):
                                st.write(f"{i+1}. {det['class_name']} (Confidence: {det['confidence']:.2f})")
                            
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
                            st.markdown(f'<h2 style="color:{get_crowd_color(crowd_level)};">{crowd_level}</h2>', unsafe_allow_html=True)
                        
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
    # Initialize session state for page navigation if not already set
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    
    # Get current page from sidebar
    current_page = render_sidebar()
    
    # Display the selected page based on session state
    if st.session_state["page"] == "Home":
        home_page()
    elif st.session_state["page"] == "Upload & Detect":
        upload_detect_page()
    elif st.session_state["page"] == "Detection Logs":
        detection_logs_page()
    elif st.session_state["page"] == "About":
        about_page()

if __name__ == "__main__":
    main() 
