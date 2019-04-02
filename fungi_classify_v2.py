import os
import cv2
import argparse
import numpy as np
import pandas as pd
from keras import Model
from imutils import paths
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator

"""file path"""
checkpoint_path = './model/checkpoint_1.h5'
name_csv = "./dataset/train_val_annotations/train.csv"

"""parameters"""
batch_size = 32
epochs = 100

parser = argparse.ArgumentParser()
parser.add_argument("-td", "--traindir", required=True, help="path to train dir")
parser.add_argument("-vd", "--valdir", required=True, help="path to val dir")
args = parser.parse_args()


def load_images():
    print("[INFO] loading images...")
    x_train = []
    y_train = []
    x_val = []
    y_val = []

    train_image_paths = sorted(list(paths.list_images(args.traindir)))
    for train_image_path in train_image_paths:
        image = cv2.imread(train_image_path, 0)
        image = cv2.resize(image, (64, 64))
        image = np.reshape(image, (64, 64, 1))
        x_train.append(image)
        label = train_image_path.split(os.path.sep)[-2]
        label = np.reshape(label, -1)
        y_train.append(label)
    x_train = np.array(x_train, dtype="float") / 255
    y_train = np.array(y_train)

    val_images_paths = sorted(list(paths.list_images(args.valdir)))
    for image_path in val_images_paths:
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (64, 64))
        image = np.reshape(image, (64, 64, 1))
        x_val.append(image)
        label = image_path.split(os.path.sep)[-2]
        label = np.reshape(label, -1)
        y_val.append(label)
    x_val = np.array(x_val, dtype="float") / 255
    y_val = np.array(y_val)

    return x_train, y_train, x_val, y_val


def create_model(num_classes):
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
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adadelta",
                  metrics=['accuracy'])

    return model


def train_network():
    # create model
    df_train = pd.read_csv(name_csv)
    nb_classes = len(set(df_train.category_id))
    model = create_model(nb_classes)
    x_train, y_train, x_val, y_val = load_images()

    # data augmentation
    aug = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode="nearest")

    # train the network

    checkpoints = ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1, monitor='val_acc', mode='max')
    print("[INFO] training network...")
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adadelta",
                  metrics=['accuracy'])
    model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_val, y_val),
                        steps_per_epoch=len(df_train.index) // batch_size,
                        epochs=epochs,
                        callbacks=[checkpoints])

    # result
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    train_network()
