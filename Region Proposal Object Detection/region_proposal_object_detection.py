import cv2
import argparse
import numpy as np
from tensorflow.keras.applications import ResNet50, imagenet_utils
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression


def selective_search(input_img, method="fast"):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(input_img)

    if method == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rectangles = ss.process()
    return rectangles


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-m", "--method", type=str, default="fast", choices=["fast", "quality"], help="selective search method")
ap.add_argument("-c", "--conf", type=float, default=0.9, help="minimum probability")
ap.add_argument("-f", "--filter", type=str, default=None, help="comma separated list of ImageNet labels to filter on")
ap.add_argument("-s", "--save", type=str, default=None, help="path to write result")
args = vars(ap.parse_args())

label_filters = args["filter"]
if label_filters is not None:
    label_filters = label_filters.lower().split(",")

model = ResNet50(weights="imagenet")
image = cv2.imread(args["image"])
H, W = image.shape[:2]

rects = selective_search(image, method=args["method"])
proposals = []
boxes = []

for (x, y, w, h) in rects:
    if w / float(W) < 0.1 or h / float(H) < 0.1:
        continue

    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))

    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    proposals.append(roi)
    boxes.append((x, y, w, h))

proposals = np.array(proposals)
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}

for i, p in enumerate(preds):
    _, label, prob = p[0]

    if label_filters is not None and label not in label_filters:
        continue

    if prob >= args["conf"]:
        x, y, w, h = boxes[i]
        box = (x, y, x + w, y + h)
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

for label in labels.keys():
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)

    for start_x, start_y, end_x, end_y in boxes:
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        cv2.putText(image, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

if args["save"] is not None:
    cv2.imwrite(args["save"], image)
