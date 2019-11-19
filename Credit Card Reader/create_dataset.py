import os
import cv2
import random
import numpy as np

path = "C:/Users/ErenS/Desktop/cc/"


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None, 0


def digit_augmentation(frame, dim=32):
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    random_num = np.random.randint(0, 9)

    if random_num % 2 == 0:
        frame = add_noise(frame)
    if random_num % 3 == 0:
        frame = pixelate(frame)
    if random_num % 2 == 0:
        frame = stretch(frame)
    frame = cv2.resize(frame, (dim, dim), interpolation=cv2.INTER_AREA)

    return frame


def add_noise(image):
    prob = random.uniform(0.01, 0.05)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy = image.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 1
    return noisy


def pixelate(image):
    dim = np.random.randint(8, 12)
    image = cv2.resize(image, (dim, dim), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    return image


def stretch(image):
    ran = np.random.randint(0, 3) * 2
    if np.random.randint(0, 2) == 0:
        frame = cv2.resize(image, (32, ran + 32), interpolation=cv2.INTER_AREA)
        return frame[int(ran / 2):int(ran + 32) - int(ran / 2), 0:32]
    else:
        frame = cv2.resize(image, (ran + 32, 32), interpolation=cv2.INTER_AREA)
        return frame[0:32, int(ran / 2):int(ran + 32) - int(ran / 2)]


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


cc1 = cv2.imread(path + "cc1_digits.jpg", 0)
cv2.imshow("cc1", cc1)
cv2.waitKey(0)

cc2 = cv2.imread(path + "cc2_digits.jpg", 0)
cv2.imshow("cc2", cc2)
cv2.waitKey(0)

_, th2 = cv2.threshold(cc2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("th2", th2)
cv2.waitKey(0)

for i in range(0, 10):
    directory_name = path + "train/" + str(i)
    print(directory_name)
    makedir(directory_name)

for i in range(0, 10):
    directory_name = path + "test/" + str(i)
    print(directory_name)
    makedir(directory_name)

cc1 = cv2.imread(path + "cc1_digits.jpg", 0)
_, th2 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("th2", th2)
cv2.waitKey(0)
cv2.imshow("cc1", cc1)
cv2.waitKey(0)

region = [(2, 19), (50, 72)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

for i in range(0, 10):
    if i > 0:
        top_left_x = top_left_x + 59
        bottom_right_x = bottom_right_x + 59

    roi = cc1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Digit - ", str(i))

    for j in range(0, 2000):
        roi2 = digit_augmentation(roi)
        roi_otsu = pre_process(roi2, inv=True)
        cv2.imwrite(path + "train/" + str(i) + "./_1_" + str(j) + ".jpg", roi_otsu)

cc1 = cv2.imread(path + "cc2_digits.jpg", 0)
_, th2 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("th2", th2)
cv2.waitKey(0)

region = [(0, 0), (35, 48)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

for i in range(0, 10):
    if i > 0:
        top_left_x = top_left_x + 35
        bottom_right_x = bottom_right_x + 35

    roi = cc1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Digit - ", str(i))
    for j in range(0, 2000):
        roi2 = digit_augmentation(roi)
        roi_otsu = pre_process(roi2, inv=False)
        cv2.imwrite(path + "train/" + str(i) + "./_2_" + str(j) + ".jpg", roi_otsu)

cc1 = cv2.imread(path + "cc1_digits.jpg", 0)
_, th2 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("th2", th2)
cv2.waitKey(0)
cv2.imshow("cc1", cc1)
cv2.waitKey(0)

region = [(2, 19), (50, 72)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

for i in range(0, 10):
    if i > 0:
        top_left_x = top_left_x + 59
        bottom_right_x = bottom_right_x + 59

    roi = cc1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Digit - ", str(i))
    for j in range(0, 200):
        roi2 = digit_augmentation(roi)
        roi_otsu = pre_process(roi2, inv=True)
        cv2.imwrite(path + "test/" + str(i) + "./_1_" + str(j) + ".jpg", roi_otsu)

cc1 = cv2.imread(path + "cc2_digits.jpg", 0)
_, th2 = cv2.threshold(cc1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("th2", th2)
cv2.waitKey(0)

region = [(0, 0), (35, 48)]

top_left_y = region[0][1]
bottom_right_y = region[1][1]
top_left_x = region[0][0]
bottom_right_x = region[1][0]

for i in range(0, 10):
    if i > 0:
        top_left_x = top_left_x + 35
        bottom_right_x = bottom_right_x + 35

    roi = cc1[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    print("Augmenting Digit - ", str(i))
    for j in range(0, 200):
        roi2 = digit_augmentation(roi)
        roi_otsu = pre_process(roi2, inv=False)
        cv2.imwrite(path + "test/" + str(i) + "./_2_" + str(j) + ".jpg", roi_otsu)

import os
import cv2
import numpy as np

path = "C:/Users/ErenS/Desktop/cc/"

trainImgs = []
trainLabels = []
testImgs = []
testLabels = []

for mainDir in os.listdir(path):
    if mainDir == "train":
        for dir in os.listdir(path + mainDir):
            print(dir)
            for file in os.listdir(path + mainDir + "/" + dir):
                img = cv2.imread(path + mainDir + "/" + dir + "/" + file)
                trainImgs.append(img)
                trainLabels.append(int(dir))
    elif mainDir == "test":
        for dir in os.listdir(path + mainDir):
            print(dir)
            for file in os.listdir(path + mainDir + "/" + dir):
                img = cv2.imread(path + mainDir + "/" + dir + "/" + file)
                testImgs.append(img)
                testLabels.append(int(dir))

np.savez(path + "train.npz", np.array(trainImgs))
np.savez(path + "train_label.npz", np.array(trainLabels))
np.savez(path + "test.npz", np.array(testImgs))
np.savez(path + "test_label.npz", np.array(testLabels))
