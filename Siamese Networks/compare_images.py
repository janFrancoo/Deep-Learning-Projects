import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model")

image_a = cv2.imread("test/image_01.png", 0)
image_b = cv2.imread("test/image_11.png", 0)

image_a = np.expand_dims(image_a, axis=-1)
image_b = np.expand_dims(image_b, axis=-1)

image_a = np.expand_dims(image_a, axis=0)
image_b = np.expand_dims(image_b, axis=0)

image_a = image_a / 255.0
image_b = image_b / 255.0

predictions = model.predict([image_a, image_b])
probability = predictions[0][0]
print(probability)
