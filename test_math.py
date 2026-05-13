import numpy as np
from PIL import Image

def test_image(image_path):
    print(f"\n--- Testing {image_path} ---")
    img = Image.open(image_path).convert('RGB').resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    
    max_c = np.max(arr, axis=-1)
    min_c = np.min(arr, axis=-1)
    diff = max_c - min_c

    exg = 2 * g - r - b
    exr = 1.4 * r - g

    is_green = (exg > 0.1) & (diff > 0.1)
    is_brown = (exr > 0.15) & (r > g) & (g > b) & (diff > 0.25)

    valid_pixels = is_green | is_brown
    leaf_percentage = np.mean(valid_pixels)

    print(f"Total Valid Pixels: {np.sum(valid_pixels)} out of {64*64}")
    print(f"Leaf Percentage: {leaf_percentage:.4f}")
    print(f"Is Green Pixels: {np.sum(is_green)}")
    print(f"Is Brown Pixels: {np.sum(is_brown)}")
    print(f"Passes Threshold (0.04)? {leaf_percentage >= 0.04}")

test_image('media/WhatsApp Image 2026-03-18 at 5.01.19 PM.jpeg')
