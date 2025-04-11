# Crowd Monitoring Module

The Crowd Monitoring module is a feature of Rail Raksha that uses computer vision to analyze crowd density in railway stations and platforms. It helps railway authorities monitor and manage passenger flow, identify potential overcrowding situations, and take appropriate action.

## Features

- **People Detection**: Uses YOLO object detection to identify people in images
- **Crowd Classification**: Classifies crowd levels as "Safe", "Moderate", or "Overcrowded" based on the number of people detected
- **Visual Feedback**: Provides annotated images with bounding boxes and crowd status labels
- **Data Logging**: Records detection results (timestamp, people count, crowd level) for historical analysis
- **Webcam Integration**: Supports real-time analysis using webcam feed
- **Statistical Analysis**: Provides charts and metrics for crowd density trends

## How It Works

1. **Image Input**: The module accepts images from file upload or webcam capture
2. **People Detection**: YOLO model detects people in the image
3. **Crowd Classification**: The number of people is used to classify the crowd level:
   - **Safe**: 0-10 people
   - **Moderate**: 11-30 people
   - **Overcrowded**: 31+ people
4. **Visual Annotation**: The image is annotated with bounding boxes and a crowd status label
5. **Data Logging**: Detection results are logged to `data/detections.csv`
6. **Analysis**: Historical data is analyzed to provide insights on crowd patterns

## Usage

1. Navigate to the "Crowd Monitoring" page in the Rail Raksha application
2. Upload an image or use your webcam to capture one
3. Click "Analyze Crowd" to process the image
4. View the results, including people count and crowd level
5. Check the "Crowd Detection History" section for historical data and trends

## Configuration

- **Confidence Threshold**: Adjust the detection confidence threshold (default: 0.25)
- **Custom Model**: Option to use a custom YOLO model for people detection

## Data Format

The crowd detection data is stored in `data/detections.csv` with the following columns:

- **timestamp**: Date and time of the detection
- **image_filename**: Name of the analyzed image
- **people_count**: Number of people detected
- **crowd_level**: Classification of the crowd level (Safe, Moderate, Overcrowded)

## Integration

The Crowd Monitoring module is integrated with the main Rail Raksha application and can be accessed from the sidebar navigation. It uses the same YOLO model infrastructure as the main object detection feature but focuses specifically on people detection and crowd analysis. 