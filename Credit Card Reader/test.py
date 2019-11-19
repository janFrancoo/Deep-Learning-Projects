import cv2
import keras
import numpy as np
from keras.models import load_model

path = "C:/Users/ErenS/Desktop/cc/"
region = [(55, 210), (640, 290)]
classifier = load_model(path + "cc.h5")


def pre_process(image, inv=False):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray_image = image
        pass

    if not inv:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(th2, (32, 32), interpolation=cv2.INTER_AREA)
    return resized


def x_cord_contour(contours):
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10'] / M['m00']))
    else:
        pass


img = cv2.imread(path + "credit_card_extracted_digits.jpg")
orig_img = cv2.imread(path + "credit_card_color.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image", img)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)
edged = cv2.Canny(blurred, 30, 150)
cv2.imshow("edged", edged)
cv2.waitKey(0)
_, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:13]
contours = sorted(contours, key=x_cord_contour, reverse=False)

full_number = []

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    if w >= 5 and h >= 25 and cv2.contourArea(c) < 1000:
        roi = blurred[y:y + h, x:x + w]
        roi_otsu = pre_process(roi, True)
        roi_otsu = cv2.cvtColor(roi_otsu, cv2.COLOR_GRAY2RGB)
        roi_otsu = keras.preprocessing.image.img_to_array(roi_otsu)
        roi_otsu = roi_otsu * 1. / 255
        roi_otsu = np.expand_dims(roi_otsu, axis=0)
        image = np.vstack([roi_otsu])
        label = str(classifier.predict_classes(image, batch_size=10))[1]
        print(label)
        (x, y, w, h) = (x + region[0][0], y + region[0][1], w, h)
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(orig_img, label, (x, y + 90), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", orig_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
