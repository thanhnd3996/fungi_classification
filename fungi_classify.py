import pandas as pd
from keras import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator

"""file path"""
checkpoint_path = './model/checkpoint.h5'
model_path = './model/model.h5'
train_dir = "./dataset/train_images"
val_dir = "./dataset/val_images"


def create_model(num_classes):
    # create a resnet pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False)

    # first, train only the top layers
    # freeze all convolutional resnet50 layers
    for layer in base_model.layers:
        layer.trainable = False

    # add a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add a fully-connected layer
    x = Dense(2048, activation='relu')(x)

    # add a final logistic layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # model
    model = Model(inputs=base_model.input, outputs=predictions)
    # model.summary()

    return model


def augment_data(model, epochs=100, batch_size=32,
                 img_width=64, img_height=64,
                 nb_train_samples=85578, nb_val_samples=4182):
    train_data_gen = ImageDataGenerator(1. / 255,
                                        horizontal_flip=True,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        rotation_range=20)
    val_data_gen = ImageDataGenerator(rescale=1. / 255)

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
    model.compile(loss='sparse_categorical_crossentropy',
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

    # show accuracy
    score = model.evaluate_generator(train_generator, verbose=0)
    print('Test score: ', score[0])
    print('Test accuracy: ', score[1])


def train_model(train_df):
    nb_classes = len(set(train_df.category_id))
    print("Creating model...")
    model = create_model(nb_classes)
    augment_data(model)


if __name__ == '__main__':
    df_train = pd.read_csv("./dataset/train_val_annotations/train.csv")
    train_model(df_train)
