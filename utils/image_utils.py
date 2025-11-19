import cv2
import numpy as np
from pathlib import Path

def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(str(image_path))

def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image

def save_processed_image(image: np.ndarray, output_path: str) -> None:
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image)
