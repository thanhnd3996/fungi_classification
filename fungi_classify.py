from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from imutils import paths
import numpy as np
import argparse
import pickle
import random
import cv2
import os

# parameters
classes = 1394
batch_size = 128
pool_size = (2, 2)
kernel_size = (3, 3)
epochs = 20
data_augmentation = True
file_path = './checkpoints/model/h5'

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset of images")
parser.add_argument("-m", "--model", required=True,
                    help="path to output trained model")
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
checkpoints = ModelCheckpoint(file_path, save_best_only=True, verbose=1, monitor='val_acc', mode='max')
model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    epochs=epochs,
                    verbose=1,
                    callbacks=[checkpoints])

# result
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# save the model and label binarizer to disk
model.save(args.model)
f = open(args.labelbin, 'wb')
f.write(pickle.dumps(lb))
f.close()

"""
[INFO] loading images...
1.WARNING:tensorflow:
From /home/lab802/miniconda3/envs/fungi/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: 
colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating: Colocations handled automatically by placer.
2./home/lab802/miniconda3/envs/fungi/lib/python3.7/site-packages/keras_applications/resnet50.py:265: 
UserWarning: The output shape of `ResNet50(include_top=False)` has been changed since Keras 2.2.0.
warnings.warn('The output shape of `ResNet50(include_top=False)` '
3.WARNING:tensorflow:
From /home/lab802/miniconda3/envs/fungi/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066:
to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version. 
Instructions for updating:Use tf.cast instead.
[INFO] training network...
WARNING:tensorflow:
From /home/lab802/miniconda3/envs/fungi/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 
(from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Traceback (most recent call last):
  File "model.py", line 80, in <module>
    callbacks=[checkpoints])
  File "/home/lab802/miniconda3/envs/fungi/lib/python3.7/site-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "/home/lab802/miniconda3/envs/fungi/lib/python3.7/site-packages/keras/engine/training.py", 
    line 1418, in fit_generator: initial_epoch=initial_epoch)
  File "/home/lab802/miniconda3/envs/fungi/lib/python3.7/site-packages/keras/engine/training_generator.py", line 55, 
    in fit_generator: raise ValueError('`steps_per_epoch=None` is only valid for a'
ValueError: `steps_per_epoch=None` is only valid for a generator based on the `keras.utils.Sequence` class. 
Please specify `steps_per_epoch` or use the `keras.utils.Sequence` class.
"""
