from PIL import Image, ImageFilter
import numpy as np

def imageToInput(image_path: str) -> np.ndarray:
    # Open the image file
    with Image.open(image_path) as img:
        img.show()
        # Resize the image to 28x28 pixels
        img = img.resize((28, 28), Image.LANCZOS)
        # Convert the image to grayscale
        img = img.convert('L')
        img = img.filter(ImageFilter.SHARPEN)
        img.show()
        # Convert the image to a numpy array
        img_array = (255 - np.array(img).flatten()) / 255.0 * 0.99 + 0.01
        return img_array