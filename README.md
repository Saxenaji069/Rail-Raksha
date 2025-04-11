# Rail Raksha

Rail Raksha is a Streamlit-based web application for railway infrastructure monitoring using computer vision and deep learning.

## Project Structure

```
Rail Raksha/
├── app.py              # Main Streamlit application
├── models/             # ML model implementations
│   ├── detector.py     # YOLO-based object detector
│   └── yolo_model.py   # YOLO model wrapper
├── utils/              # Utility functions
│   ├── image_utils.py  # Image processing utilities
│   └── data_logger.py  # Data logging utilities
├── uploads/            # Directory for uploaded files
├── data/              # Mock data and logs
│   └── detections.csv # Detection log file
└── requirements.txt    # Project dependencies
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Features

- Image and video upload support
- Real-time object detection using YOLO
- Infrastructure defect detection
- Detection history logging and analysis
- Interactive data visualization
- Easy-to-use web interface

## Dependencies

- Streamlit
- OpenCV
- PyTorch
- Ultralytics (YOLO)
- NumPy
- Pandas
- Plotly

## License

MIT License 