import cv2
import numpy as np
from tensorflow.keras.models import load_model

classes = ['Positive', 'Negative']
model = load_model("C:/Users/PC/Documents/Data/covid-19/weights.h5")

test_image_1 = cv2.imread("C:/Users/PC/Documents/Data/covid-19/covid/ryct.2020200034.fig5-day0.jpeg")
test_image_2 = cv2.imread("C:/Users/PC/Documents/Data/covid-19/normal/person934_virus_1595.jpeg")

test_image_1_original = test_image_1
test_image_1_original = cv2.resize(test_image_1_original, (722, 722))
test_image_2_original = test_image_2
test_image_2_original = cv2.resize(test_image_2_original, (722, 722))

test_image_1 = cv2.cvtColor(test_image_1, cv2.COLOR_BGR2RGB)
test_image_1 = cv2.resize(test_image_1, (224, 224))
test_image_1 = test_image_1.astype("float32")

test_image_2 = cv2.cvtColor(test_image_2, cv2.COLOR_BGR2RGB)
test_image_2 = cv2.resize(test_image_2, (224, 224))
test_image_2 = test_image_2.astype("float32")

prediction = model.predict(np.expand_dims(test_image_1, axis=0))[0]
print(prediction)
label = classes[np.argmax(prediction)]
cv2.putText(test_image_1_original, label + "!", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

cv2.imshow("Result", test_image_1_original)
cv2.waitKey()

prediction = model.predict(np.expand_dims(test_image_2, axis=0))[0]
print(prediction)
label = classes[np.argmax(prediction)]
cv2.putText(test_image_2_original, label + "!", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

cv2.imshow("Result", test_image_2_original)
cv2.waitKey()
