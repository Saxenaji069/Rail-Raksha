import cv2
import numpy as np
from pathlib import Path

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the given path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Loaded image
    """
    return cv2.imread(str(image_path))

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for model input.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Preprocessed image
    """
    # Resize image to standard size
    image = cv2.resize(image, (640, 640))
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize
    image = image / 255.0
    
    return image

def save_processed_image(image: np.ndarray, output_path: str) -> None:
    """
    Save the processed image to the specified path.
    
    Args:
        image (np.ndarray): Image to save
        output_path (str): Path where to save the image
    """
    # Convert back to uint8
    image = (image * 255).astype(np.uint8)
    
    # Convert back to BGR for saving
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(output_path), image) 