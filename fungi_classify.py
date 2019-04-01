<<<<<<< HEAD
import pandas as pd
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

"""file path"""
checkpoint_1_path = '../model/checkpoint_1.h5'
checkpoint_2_path = '../model/checkpoint_2.h5'
model_json_path = '../model/model.json'
model_1_path = '../model/model_1.h5'
model_2_path = '../model/model_2.h5'
train_dir = ""
val_dir = ""


def create_model(num_classes, model_file):
    # create a resnet pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False)

    # add a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add a fully-connected layer
    x = Dense(2048, activation='relu')(x)

    # add a final logistic layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # model
    model = Model(inputs=base_model.input, outputs=predictions)

    # first, train only the top layers
    # freeze all convolutional resnet50 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    print("[INFO] training network...")
    model.compile(loss="categorical_crossentropy",
                  optimizer="adadelta",
                  metrics=['accuracy'])

    # serialize model to json
    model_json = model.to_json()
    with open(model_file, 'w') as f:
        f.write(model_json)

    return model


def fine_tuning(model, epochs_1=15, epochs_2=30, patience_1=1, patience_2=1, batch_size=32,
                img_width=229, img_height=229, nb_train_samples=85578, nb_val_samples=4182):
    train_data_gen = ImageDataGenerator(preprocessing_function=pre_process_input,
                                        horizontal_flip=True,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        rotation_range=20)
    val_data_gen = ImageDataGenerator(preprocessing_function=pre_process_input)

    # define train & val data generators
    train_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_data_gen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs_1,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience_1),
                   ModelCheckpoint(filepath=checkpoint_1_path, verbose=1, save_best_only=True)])

    model.save_weights(model_1_path)

    # we chose to train the top 2 resnet blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs_2,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience_2),
                   ModelCheckpoint(filepath=checkpoint_2_path, verbose=1, save_best_only=True)])

    # save final weights
    model.save_weights(model_2_path)


def pre_process_input(x):
    # scaling
    x /= 255
    x -= 0.5
    x *= 2
    return x


def train_model(train_df, val_df):
    nb_classes = len(set(train_df.category_id))
    print("Creating model...")
    model = create_model(nb_classes, model_file=model_json_path)
    print("Fine-tuning model...")
    fine_tuning(model, epochs_1=15, epochs_2=30, patience_1=1, patience_2=2, batch_size=512,
                nb_train_samples=len(train_df.index), nb_val_samples=len(val_df.index))


if __name__ == '__main__':
    df_train = pd.read_csv("../dataset/train_val_annotations/train.csv")
    df_val = pd.read_csv("../dataset/train_val_annotations/val.csv")
    train_model(df_train, df_val)
=======
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from model.resnet import resnet50
from keras.callbacks import ModelCheckpoint
from imutils import paths
import numpy as np
import argparse
import json
import random
import cv2
import os

# parameters
batch_size = 32
epochs = 1000

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset of images")
parser.add_argument("-m", "--model", required=True,
                    help="path to best trained model ")
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
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = np.reshape(image, (64, 64, 1))
    data.append(image)
    label = image_path.split(os.path.sep)[-2]
    label = np.reshape(label, -1)
    labels.append(label)

# scale the raw pixel intensities to the range [0,1]
data = np.array(data, dtype='float') / 255
labels = np.array(labels)

# partition the data into training and validation
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.04182, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# init the model and optimizer
model = resnet50((64, 64, 1), len(lb.classes_))

# model = ResNet50(weights='imagenet', include_top=False)
print("[INFO] training network...")
model.compile(loss="parse_categorical_crossentropy",
              optimizer="adadelta",
              metrics=['accuracy'])

# train and save the model
model = resnet50((64, 64, 1), len(lb.classes_))

# model = ResNet50(weights='imagenet', include_top=False)
print("[INFO] training network...")
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adadelta",
              metrics=['accuracy'])

# train the network
checkpoints = ModelCheckpoint(args.model, save_best_only=True, verbose=1, monitor='val_acc', mode='max')
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
f.write(json.dumps(lb))
f.close()
>>>>>>> 96f596d4b554d670fd3f8f7bb7e5d9f1093d6651
