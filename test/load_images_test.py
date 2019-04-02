import os
from imutils import paths
import cv2
import numpy as np

val_dir = "../dataset/val_images/"

val_data = []
val_labels = []
image_paths = sorted(list(paths.list_images(val_dir)))
for image_path in image_paths:
    print(image_path)
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (64, 64))
    image = np.reshape(image, (64, 64, 1))
    val_data.append(image)
    label = image_path.split(os.path.sep)[-2]
    # print(label)
    val_labels.append(label)
    print(val_labels)
