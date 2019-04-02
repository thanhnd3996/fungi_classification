import pandas as pd
from keras import Model
from data.load_images import load_images
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator

"""file path"""
checkpoint_path = './model/checkpoint_1.h5'
model_path = './model/model.h5'
name_csv = "./dataset/train_val_annotations/train.csv"

"""parameters"""
batch_size = 32
epochs = 100


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
    model.compile(loss="categorical_crossentropy",
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
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=epochs,
                        callbacks=[checkpoints])

    # result
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if '__name__' == '__main__':
    train_network()
