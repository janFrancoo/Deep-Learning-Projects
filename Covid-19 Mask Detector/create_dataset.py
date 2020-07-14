import os
import cv2


def get_data(path):
    images = []
    labels = []
    for file_name in os.listdir(path):
        images.append(cv2.imread(os.path.join(path, file_name)))
        if file_name.startswith("masked"):
            labels.append(1)
        else:
            labels.append(0)
    return images, labels
