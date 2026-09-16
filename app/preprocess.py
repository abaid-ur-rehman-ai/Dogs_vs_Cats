import cv2
import numpy as np

def preprocess_image(file_bytes):
    # Convert the raw uploaded bytes into a NumPy array
    np_array = np.frombuffer(file_bytes, np.uint8)

    # Decode that array into an actual image (like cv2.imread, but from memory)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # add right after cv2.imdecode

    # Resize to match what the model was trained on
    img = cv2.resize(img, (256, 256))

    # Reshape to add the "batch" dimension: (1, 256, 256, 3)
    img = img.reshape(1, 256, 256, 3)

    # Normalize pixel values from 0-255 down to 0-1
    img = img / 255.0

    return img