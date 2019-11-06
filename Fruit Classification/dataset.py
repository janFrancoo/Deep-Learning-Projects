import os
import cv2
import numpy as np

path = "C:/Users/ErenS/Desktop/fruits-360/"

trainImgs = []
trainLabels = []
testImgs = []
testLabels = []

for mainDir in os.listdir(path):
    if mainDir == "train":
        for dir in os.listdir(path + mainDir):
            splittedLabel = dir.split(" ")
            label = ""
            length = len(splittedLabel)
            for i in range(length):
                if not splittedLabel[i].isnumeric():
                    label += splittedLabel[i] + " "
            print(label)
            for file in os.listdir(path + mainDir + "/" + dir):
                img = cv2.imread(path + mainDir + "/" + dir + "/" + file)
                img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
                trainImgs.append(img)
                trainLabels.append(label)
    elif mainDir == "validation":
        for dir in os.listdir(path + mainDir):
            splittedLabel = dir.split(" ")
            label = ""
            length = len(splittedLabel)
            for i in range(length):
                if not splittedLabel[i].isnumeric():
                    label += splittedLabel[i] + " "
            print(label)
            for file in os.listdir(path + mainDir + "/" + dir):
                img = cv2.imread(path + mainDir + "/" + dir + "/" + file)
                img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
                testImgs.append(img)
                testLabels.append(label)

np.savez("train.npz", np.array(trainImgs))
np.savez("train_label.npz", np.array(trainLabels))
np.savez("test.npz", np.array(testImgs))
np.savez("test_label.npz", np.array(testLabels))
