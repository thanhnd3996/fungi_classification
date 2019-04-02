import os
import cv2
import numpy as np
from imutils import paths

train_dir = "./dataset/train_images"
val_dir = "./dataset/val_images"


def loop_over_input_images(path):
    list_data = []
    list_label = []
    image_paths = sorted(list(paths.list_images(path)))
    for image_path in image_paths:
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (64, 64))
        image = np.reshape(image, (64, 64, 1))
        list_data.append(image)
        label = image_path.split(os.path.sep)[-2]
        label = np.reshape(label, -1)
        list_label.append(label)
    return list_data, list_label


def load_images():
    print("[INFO] loading images...")
    train_images_paths = sorted(list(paths.list_images(train_dir)))
    val_images_paths = sorted(list(paths.list_images(val_dir)))
    x_train, y_train = loop_over_input_images(train_images_paths)
    x_train = np.array(x_train, dtype="float") / 255
    y_train = np.array(y_train)
    x_val, y_val = loop_over_input_images(val_images_paths)
    return x_train, y_train, x_val, y_val
