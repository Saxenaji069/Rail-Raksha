import random
import cv2
import numpy as np
from pathlib import Path

def detect_crime(image_path):
    """
    Mock function to detect crime or suspicious behavior in an image.
    Randomly returns "no crime" or "suspicious behavior".
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Detection results with status and confidence
    """
    # Load the image to ensure it exists
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Randomly decide if there's a crime
    is_crime = random.random() < 0.3  # 30% chance of crime
    
    if is_crime:
        # Randomly select a crime type
        crime_types = [
            "suspicious behavior", 
            "trespassing", 
            "vandalism", 
            "loitering"
        ]
        crime_type = random.choice(crime_types)
        confidence = random.uniform(0.6, 0.95)
        return {
            "status": crime_type,
            "confidence": confidence,
            "location": "railway station" if random.random() < 0.5 else "railway track"
        }
    else:
        return {
            "status": "no crime",
            "confidence": random.uniform(0.8, 0.99),
            "location": "railway station" if random.random() < 0.5 else "railway track"
        }

def detect_cleanliness(image_path):
    """
    Mock function to assess cleanliness of railway infrastructure.
    Returns a score from 0 to 100.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Cleanliness assessment with score and details
    """
    # Load the image to ensure it exists
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Generate a random cleanliness score
    score = random.randint(0, 100)
    
    # Determine cleanliness level
    if score >= 90:
        level = "Excellent"
    elif score >= 70:
        level = "Good"
    elif score >= 50:
        level = "Fair"
    elif score >= 30:
        level = "Poor"
    else:
        level = "Very Poor"
    
    # Generate random issues
    issues = []
    if score < 80:
        potential_issues = [
            "litter", "graffiti", "rust", "weeds", 
            "broken equipment", "dirt", "debris"
        ]
        num_issues = random.randint(1, min(3, len(potential_issues)))
        issues = random.sample(potential_issues, num_issues)
    
    return {
        "score": score,
        "level": level,
        "issues": issues
    } 