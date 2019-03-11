from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from keras.callbacks import ModelCheckpoint
from model.resnet import resnet50
from imutils import paths
import numpy as np
import argparse
import json
=======
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from imutils import paths
import numpy as np
import argparse
import pickle
>>>>>>> d09d9a6ef3c8cb705f9e983bdfd531d32a14c7c3
import random
import cv2
import os

# parameters
<<<<<<< HEAD
batch_size = 32
epochs = 1000
=======
classes = 1394
batch_size = 128
pool_size = (2, 2)
kernel_size = (3, 3)
epochs = 20
>>>>>>> d09d9a6ef3c8cb705f9e983bdfd531d32a14c7c3

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset of images")
parser.add_argument("-l", "--labelbin", required=True,
                    help="path to output label binarizer")
args = parser.parse_args()

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(args.dataset)))
random.seed(42)
random.shuffle(image_paths)

# loop over the input images
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
<<<<<<< HEAD
    image = np.reshape(image, (64, 64, 1))
    data.append(image)
    label = image_path.split(os.path.sep)[-2]
    label = np.reshape(label, -1)
=======
    data.append(image)
    label = image_path.split(os.path.sep)[-2]
>>>>>>> d09d9a6ef3c8cb705f9e983bdfd531d32a14c7c3
    labels.append(label)

# scale the raw pixel intensities to the range [0,1]
data = np.array(data, dtype='float') / 255
labels = np.array(labels)

# partition the data into training and validation
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.04182, random_state=42)
<<<<<<< HEAD
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
=======
>>>>>>> d09d9a6ef3c8cb705f9e983bdfd531d32a14c7c3

# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# init the model and optimizer
<<<<<<< HEAD
model = resnet50((64, 64, 1), len(lb.classes_))

# model = ResNet50(weights='imagenet', include_top=False)
print("[INFO] training network...")
model.compile(loss="parse_categorical_crossentropy",
              optimizer="adadelta",
              metrics=['accuracy'])

# train and save the model
file_path = './output/model.h5'
=======
base_model = ResNet50(weights='imagenet', include_top=False)
train_features = base_model.predict(x_train)
model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=train_features))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

# model = ResNet50(weights='imagenet', include_top=False)
print("[INFO] training network...")
model.compile(loss="categorical_crossentropy",
              optimizer="adadelta",
              metrics=['accuracy'])

# train the network
file_path = './checkpoints/model/h5'
>>>>>>> d09d9a6ef3c8cb705f9e983bdfd531d32a14c7c3
checkpoints = ModelCheckpoint(file_path, save_best_only=True, verbose=1, monitor='val_acc', mode='max')
model.fit(x_train, y_train, batch_size=batch_size,
          validation_data=(x_test, y_test),
          epochs=epochs,
          verbose=1,
          callbacks=[checkpoints])

# result
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# label binarizer to disk
f = open(args.labelbin, 'wb')
<<<<<<< HEAD
f.write(json.dumps(lb))
=======
f.write(pickle.dumps(lb))
>>>>>>> d09d9a6ef3c8cb705f9e983bdfd531d32a14c7c3
f.close()
