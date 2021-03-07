import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

with open("test_images.txt", "r") as file:
    lines = file.readlines()
    image_path = lines[np.random.randint(0, len(lines))].strip("\n")

model = load_model("detector.h5")
label_binarizer = pickle.loads(open("label_binarizer.pickle", "rb").read())

image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

box_preds, label_preds = model.predict(image)
start_x, start_y, end_x, end_y = box_preds[0]

i = np.argmax(label_preds, axis=1)
label = label_binarizer.classes_[i][0]

image = cv2.imread(image_path)
h, w = image.shape[:2]

start_x = int(start_x * w)
start_y = int(start_y * h)
end_x = int(end_x * w)
end_y = int(end_y * h)

y = start_y - 10 if start_y - 10 > 10 else start_y + 10
cv2.putText(image, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

cv2.imshow("res", image)
cv2.waitKey(0)
