import cv2
import numpy as np
from keras.models import load_model
from google.colab.patches import cv2_imshow

classifier = load_model('drive/ColabNotebooks/catsvsdogs.h5')

def draw_test(name, pred, inputImg):
  BLACK = [0, 0, 0]
  if pred == "[0]":
    pred = "cat"
  if pred == "[1]":
    pred = "dog"
  expandedImg = cv2.copyMakeBorder(inputImg, 0, 0, 0, imageL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)
  cv2.putText(expandedImg, str(pred), (252, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
  cv2_imshow(expandedImg * 255)
  
for i in range(10):
  rand = np.random.randint(0, len(x_test))
  inputImg = x_test[rand]
  
  imageL = cv2.resize(inputImg, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
  cv2_imshow(imageL)
  
  inputImg = inputImg.reshape(1, 150, 150, 3)
  
  res = str(classifier.predict_classes(inputImg, 1, verbose = 0)[0])
  
  draw_test("Prediction", res, imageL)
