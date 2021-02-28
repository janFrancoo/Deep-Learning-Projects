import numpy as np


def make_pairs(images, labels):
    num_classes = len(np.unique(labels))
    indexes_of_labels = [np.where(labels == x)[0] for x in range(0, num_classes)]

    pair_images = []
    pair_labels = []
    for image, label in zip(images, labels):
        pos_idx = np.random.choice(indexes_of_labels[label])
        pos_image = images[pos_idx]

        pair_images.append([image, pos_image])
        pair_labels.append([1])

        neg_idx = np.random.choice(np.where(labels != label)[0])
        neg_image = images[neg_idx]

        pair_images.append([image, neg_image])
        pair_labels.append([0])

    return np.array(pair_images), np.array(pair_labels)
