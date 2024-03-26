import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
loaded_model = tf.keras.models.load_model(r'E:\coding\python\fakeimage\my_strong_model.h5')

# Specify the image path
image_path = r"D:\Pictures\DSC_5116.jpg"

# Load and preprocess the image
img = cv2.imread(image_path)

# Check if the image is loaded successfully
if img is None or img.size == 0:
    print(f"Error: Unable to read the image at {image_path}")
else:
    # Resize the image to the input size of the model
    img = cv2.resize(img, (32, 32))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.reshape(img, (1, 32, 32, 3))  # Reshape to match the input shape of the model

    # Make predictions
    predictions = loaded_model.predict(img)

    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Print the result
    if predicted_class == 0:
        print("The image is predicted to be real.")
    else:
        print("The image is predicted to be fake.")
        print(predicted_class)