from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os

# parameters
learning_rate = 0.01
classes = 1394
batch_size = 128
pool_size = (2, 2)
kernel_size = (3, 3)
epochs = 20
data_augmentation = True
file_path = './checkpoints/model/h5'
checkpoints = ModelCheckpoint(file_path, save_best_only=True, verbose=1, monitor='val_acc', mode='max')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args)

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(image_paths)

# loop over the input images
for image_path in image_paths:
    image = cv2.imread(image_paths)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0,1]
data = np.array(data, dtype='float') / 255
labels = np.array(labels)

# partition the data into training and validation
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.04182, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# init the model and optimizer
model = ResNet50(weights='imagenet', include_top=False)
print("[INFO] training network...")
model.compile(loss="categorical_crossentropy",
              optimizer="adadelta",
              metrics=['accuracy'])

# train the network
model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    epochs=epochs,
                    verbose=1,
                    callbacks=[checkpoints])

# result
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
