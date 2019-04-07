# import pandas as pd
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

"""file path"""
checkpoint_path = './model/checkpoint.h5'
model_path = './model/model.h5'
train_dir = "./dataset/train_image"
val_dir = "./dataset/val_image"


def create_model(num_classes):
    # create a resnet pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

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

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop')

    model.summary()

    return model


def augment_data(model, nb_train_samples, nb_val_samples, epochs=100, batch_size=16,
                 img_width=299, img_height=299):
    train_data_gen = ImageDataGenerator(preprocessing_function=pre_process,
                                        horizontal_flip=True,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        rotation_range=20)
    val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)

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

    # compile model and fit
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        callbacks=[ModelCheckpoint(checkpoint_path,
                                   save_best_only=True, verbose=1, monitor='val_acc', mode='max')])

    model.save_weights(model_path)


def pre_process(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def train_model():
    # nb_classes = len(set(train_df.category_id))
    nb_classes = 100
    nb_train_samples = 7963
    nb_val_samples = 300
    print("Creating model...")
    model = create_model(nb_classes)
    augment_data(model, nb_train_samples, nb_val_samples)
    # nb_train_samples=len(train_df.index),
    # nb_val_samples=len(val_df.index))


if __name__ == '__main__':
    # df_train = pd.read_csv("./dataset/train_val_annotations/train.csv")
    # df_val = pd.read_csv("./dataset/train_val_annotations/val.csv")
    # train_model(df_train, df_val)
    train_model()
