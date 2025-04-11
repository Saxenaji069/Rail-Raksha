import cv2
import numpy as np
import os
from pathlib import Path
import random
from PIL import Image, ImageDraw

def generate_fake_crowd_image(num_people, output_dir="uploads"):
    """
    Generate a fake crowd image with the specified number of people.
    
    Args:
        num_people (int): Number of people to add to the image
        output_dir (str): Directory to save the output image
        
    Returns:
        str: Path to the generated image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load a background image (empty station or park)
    # If no background image is available, create a blank one
    background_path = os.path.join(output_dir, "background.jpg")
    if os.path.exists(background_path):
        background = cv2.imread(background_path)
    else:
        # Create a blank background
        background = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Add some simple elements to make it look like a station
        cv2.rectangle(background, (100, 400), (700, 500), (200, 200, 200), -1)  # Platform
        cv2.rectangle(background, (150, 300), (650, 400), (180, 180, 180), -1)  # Wall
        
        # Save the background for future use
        cv2.imwrite(background_path, background)
    
    # Convert to PIL Image for easier manipulation
    background_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(background_pil)
    
    # Define the area where people can be placed
    usable_area = (100, 100, 700, 500)
    
    # Generate people silhouettes
    for i in range(num_people):
        # Random position within the usable area
        x = random.randint(usable_area[0], usable_area[2])
        y = random.randint(usable_area[1], usable_area[3])
        
        # Random size for variety
        size = random.randint(20, 40)
        
        # Draw a simple person silhouette
        # Head
        draw.ellipse((x-size//2, y-size//2, x+size//2, y+size//2), fill=(0, 0, 0))
        
        # Body
        draw.rectangle((x-size//4, y+size//2, x+size//4, y+size), fill=(0, 0, 0))
        
        # Legs
        draw.line((x-size//4, y+size, x-size//2, y+size+size//2), fill=(0, 0, 0), width=2)
        draw.line((x+size//4, y+size, x+size//2, y+size+size//2), fill=(0, 0, 0), width=2)
    
    # Convert back to OpenCV format
    result = cv2.cvtColor(np.array(background_pil), cv2.COLOR_RGB2BGR)
    
    # Save the image
    output_path = os.path.join(output_dir, f"test_{num_people}.jpg")
    cv2.imwrite(output_path, result)
    
    return output_path

def create_background_image(output_dir="uploads"):
    """
    Create a background image for crowd generation.
    
    Args:
        output_dir (str): Directory to save the output image
        
    Returns:
        str: Path to the generated background image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a blank background
    background = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add some simple elements to make it look like a station
    cv2.rectangle(background, (100, 400), (700, 500), (200, 200, 200), -1)  # Platform
    cv2.rectangle(background, (150, 300), (650, 400), (180, 180, 180), -1)  # Wall
    
    # Add some details
    cv2.line(background, (200, 450), (600, 450), (100, 100, 100), 2)  # Track
    cv2.line(background, (200, 460), (600, 460), (100, 100, 100), 2)  # Track
    
    # Add some signs
    cv2.rectangle(background, (300, 200), (500, 250), (0, 0, 255), -1)  # Sign
    cv2.putText(background, "STATION", (320, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save the background
    output_path = os.path.join(output_dir, "background.jpg")
    cv2.imwrite(output_path, background)
    
    return output_path 