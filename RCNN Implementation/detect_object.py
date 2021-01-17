import cv2
import pickle
import imutils
import argparse
import numpy as np
from utils import config
from utils.non_maximum_supression import non_max_suppression
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

model = load_model(config.MODEL_PATH)
label_binarizer = pickle.loads(open(config.ENCODER_PATH, "rb").read())

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

proposals = []
boxes = []
for x, y, w, h in rects[:config.MAX_PROPOSALS_INFER]:
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
    roi = img_to_array(roi)
    roi = preprocess_input(roi)
    proposals.append(roi)
    boxes.append((x, y, x + w, y + h))

proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")

prob = model.predict(proposals)
labels = label_binarizer.classes_[np.argmax(prob, axis=1)]
idxs = np.where(labels == "raccoon")[0]

boxes = boxes[idxs]
prob = prob[idxs][:, 1]
idxs = np.where(prob >= config.MIN_PROB)
boxes = boxes[idxs]
prob = prob[idxs]

clone = image.copy()
boxIdxs = non_max_suppression(boxes, prob)
for i in boxIdxs:
    start_x, start_y, end_x, end_y = boxes[i]
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    y = start_y - 10 if start_y - 10 > 10 else start_y + 10
    text = "Raccoon: {:.2f}%".format(prob[i] * 100)
    cv2.putText(image, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.imwrite("object_detected.png", image)
