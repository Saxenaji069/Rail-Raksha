import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import time
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

from utils.crowd_monitor import detect_people, log_crowd_detection, get_crowd_level

class VideoProcessor:
    """Class to handle video processing for crowd monitoring."""
    
    def __init__(self, conf_threshold=0.25, model_path=None):
        """
        Initialize the VideoProcessor.
        
        Args:
            conf_threshold (float): Confidence threshold for detections
            model_path (str, optional): Path to a custom YOLO model
        """
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.is_processing = False
        self.frame_count = 0
        self.last_log_time = time.time()
        self.log_interval = 5.0  # Log every 5 seconds
        
        # Initialize statistics
        self.reset_stats()
        
    def process_video(self, video_path: str, display_callback: Optional[Callable] = None) -> None:
        """
        Process a video file and perform object detection on each frame.
        
        Args:
            video_path (str): Path to the video file
            display_callback (Callable, optional): Callback function to display processed frames
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.is_processing = True
        cap = cv2.VideoCapture(video_path)
        
        try:
            while cap.isOpened() and self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Debug information
                print("DEBUG: Frame Shape", frame.shape)
                
                # Process frame
                try:
                    result = detect_people(
                        frame,
                        conf_threshold=self.conf_threshold
                    )
                    
                    # Handle different return value formats
                    if isinstance(result, tuple):
                        if len(result) == 5:
                            processed_frame, detections, people_count, crowd_level, violations = result
                        elif len(result) == 3:
                            processed_frame, detections, people_count = result
                            crowd_level = get_crowd_level(people_count)
                            violations = []
                        else:
                            st.error(f"Unexpected return format from detect_people")
                            continue
                    else:
                        st.error("detect_people did not return expected tuple format")
                        continue
                    
                    # Debug information
                    print("DEBUG: Detection counts", {
                        'people_count': people_count,
                        'crowd_level': crowd_level,
                        'violations': len(violations)
                    })
                    
                    # Update statistics
                    self.stats['total_frames'] += 1
                    self.total_people += people_count
                    self.stats['avg_people'] = self.total_people / self.stats['total_frames']
                    self.stats['max_people'] = max(self.stats['max_people'], people_count)
                    self.stats['crowd_levels'][crowd_level] += 1
                    
                    # Display frame if callback is provided
                    if display_callback:
                        display_callback(processed_frame)
                    
                    # Add a small delay to control processing speed
                    time.sleep(0.03)  # Approximately 30 FPS
                    
                    # Log detection at intervals
                    current_time = time.time()
                    if current_time - self.last_log_time >= self.log_interval:
                        self.log_detection(frame, processed_frame)
                        self.last_log_time = current_time
                    
                except Exception as e:
                    st.error(f"Error in detection: {str(e)}")
                    continue
                
        finally:
            cap.release()
            self.is_processing = False
    
    def get_detection_stats(self) -> Dict:
        """
        Get the statistics from video processing.
        
        Returns:
            Dict: Dictionary containing processing statistics
        """
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset all statistics to their initial values."""
        self.stats = {
            'total_frames': 0,
            'avg_people': 0,
            'max_people': 0,
            'crowd_levels': {
                'Safe': 0,
                'Moderate': 0,
                'Overcrowded': 0
            }
        }
        self.total_people = 0
    
    def log_detection(self, original_frame, processed_frame):
        """
        Log detection results.
        
        Args:
            original_frame (np.ndarray): Original frame
            processed_frame (np.ndarray): Processed frame with annotations
        """
        try:
            # Save frames temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_path = Path(f"temp_original_{timestamp}.jpg")
            processed_path = Path(f"temp_processed_{timestamp}.jpg")
            
            cv2.imwrite(str(original_path), original_frame)
            cv2.imwrite(str(processed_path), processed_frame)
            
            # Run detection to get counts - use the original frame directly
            # MODIFIED: Handle different return value formats
            try:
                result = detect_people(
                    original_frame,
                    conf_threshold=self.conf_threshold
                )
                
                if isinstance(result, tuple):
                    if len(result) == 5:
                        _, detections, people_count, crowd_level, violations = result
                    elif len(result) == 3:
                        _, detections, people_count = result
                        crowd_level = get_crowd_level(people_count)
                        violations = []
                    else:
                        st.error(f"Unexpected return format from detect_people")
                        return
                else:
                    st.error("detect_people did not return expected tuple format")
                    return
            except Exception as e:
                st.error(f"Error in detection during logging: {str(e)}")
                return
            
            # Log detection
            log_crowd_detection(
                str(original_path),
                people_count,
                crowd_level,
                violations
            )
            
            # Display detection results
            st.sidebar.markdown(f"**Frame {self.frame_count}:**")
            st.sidebar.markdown(f"- People Count: {people_count}")
            st.sidebar.markdown(f"- Crowd Level: {crowd_level}")
            if violations:
                st.sidebar.markdown(f"- Violations: {len(violations)}")
        except Exception as e:
            st.error(f"Error logging detection: {str(e)}")
        finally:
            # Clean up temporary files
            if original_path.exists():
                os.remove(original_path)
            if processed_path.exists():
                os.remove(processed_path)

    def process_frame(self, frame):
        """
        Process a single frame for crowd detection.
        
        Args:
            frame (np.ndarray): The frame to process
            
        Returns:
            np.ndarray: Processed frame with annotations
        """
        try:
            # Debug information
            print("DEBUG: Frame Shape", frame.shape)
            
            # Process frame
            result = detect_people(
                frame,
                conf_threshold=self.conf_threshold
            )
            
            # Handle different return value formats
            if isinstance(result, tuple):
                if len(result) == 5:
                    processed_frame, detections, people_count, crowd_level, violations = result
                elif len(result) == 3:
                    processed_frame, detections, people_count = result
                    crowd_level = get_crowd_level(people_count)
                    violations = []
                else:
                    st.error(f"Unexpected return format from detect_people")
                    return frame
            else:
                st.error("detect_people did not return expected tuple format")
                return frame
            
            # Debug information
            print("DEBUG: Detection counts", {
                'people_count': people_count,
                'crowd_level': crowd_level,
                'violations': len(violations)
            })
            
            # Debug information for detections
            print(f"DEBUG: {len(detections)} detections found")
            for i, det in enumerate(detections):
                print(f"Detection {i+1}: {det.get('class', 'unknown')} with confidence {det.get('confidence', 0):.2f}")
                if det.get('class', '').lower() == 'person':
                    print(f"✅ Person detected! Confidence: {det.get('confidence', 0):.2f}")
            
            # Update statistics
            self.frame_count += 1
            self.total_people += people_count
            self.stats['total_frames'] += 1
            self.stats['avg_people'] = self.total_people / self.stats['total_frames']
            self.stats['max_people'] = max(self.stats['max_people'], people_count)
            
            # Update crowd level counts
            if crowd_level in self.stats['crowd_levels']:
                self.stats['crowd_levels'][crowd_level] += 1
            
            return processed_frame
            
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            return frame

def process_webcam_feed(conf_threshold=0.25, model_path=None, display_callback=None):
    """
    Process webcam feed for crowd monitoring.
    
    Args:
        conf_threshold (float): Confidence threshold for detections
        model_path (str, optional): Path to a custom YOLO model
        display_callback (callable, optional): Function to display frames
        
    Returns:
        tuple: (VideoProcessor, stop_button)
    """
    # Initialize video processor
    processor = VideoProcessor(conf_threshold)
    
    # Reset frame count to prevent continuous counting issue
    processor.frame_count = 0
    
    # Create stop button with session state
    stop_button = st.session_state.get("stop_webcam", False) or st.button("Stop Processing", key="stop_webcam")
    
    # Create frame placeholder
    frame_placeholder = st.empty()
    
    # Define display callback if not provided
    if not display_callback:
        def display_callback(frame):
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
    
    # Initialize webcam if not already created
    if 'cap' not in locals() or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error opening webcam")
        return processor, stop_button
    
    # Set processing flag to True
    processor.is_processing = True
    
    # Process webcam feed
    try:
        while not stop_button and processor.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = processor.process_frame(frame)
            
            # Display frame
            display_callback(processed_frame)
            
            # Log detection at intervals
            current_time = time.time()
            if current_time - processor.last_log_time >= processor.log_interval:
                processor.log_detection(frame, processed_frame)
                processor.last_log_time = current_time
            
            # Add small delay to control processing speed
            time.sleep(0.05)
    finally:
        # Release resources
        cap.release()
        processor.is_processing = False
    
    return processor, stop_button