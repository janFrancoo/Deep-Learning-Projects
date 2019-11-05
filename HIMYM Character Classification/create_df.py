import os
import cv2
import numpy as np

path = "C:/Users/ErenS/Desktop/flags/faces/"

Imgs = []
Labels = []

for dir in os.listdir(path):
    print(dir)
    for file in os.listdir(path + dir):
        img = cv2.imread(path + dir + "/" + file)
        img = cv2.resize(img, (100, 100))
        Imgs.append(img)
        Labels.append(dir)

np.savez(path + "train.npz", np.array(Imgs))
np.savez(path + "train_label.npz", np.array(Labels))
