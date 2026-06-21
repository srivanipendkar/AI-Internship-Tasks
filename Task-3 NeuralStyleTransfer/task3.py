import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

print("Starting Neural Style Transfer...")

# Load image function
def load_image(path):
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    img = img[np.newaxis, :]
    return tf.constant(img)

# Save image function
def save_image(tensor, filename):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        tensor = tensor[0]

    Image.fromarray(tensor).save(filename)

# Load model
model = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
)

print("Model loaded successfully!")

# Image pairs
image_pairs = [
    ("content.jpg", "style.jpg", "output.jpg"),
    ("content1.jpg", "style1.jpg", "output1.jpg"),
    ("content2.jpg", "style2.jpg", "output2.jpg")
]

# Check files exist
print("\nChecking image files...\n")

for content_path, style_path, output_path in image_pairs:

    if os.path.exists(content_path):
        print(f"{content_path} FOUND")
    else:
        print(f"{content_path} NOT FOUND")

    if os.path.exists(style_path):
        print(f"{style_path} FOUND")
    else:
        print(f"{style_path} NOT FOUND")

print("\nStarting processing...\n")

# Process images
for content_path, style_path, output_path in image_pairs:

    try:
        print(f"Processing {content_path} + {style_path}")

        content_image = load_image(content_path)
        style_image = load_image(style_path)

        stylized_image = model(content_image, style_image)[0]

        save_image(stylized_image, output_path)

        print(f"{output_path} saved successfully!\n")

    except Exception as e:
        print(f"Error processing {content_path}: {e}")

print("All processing completed!")
