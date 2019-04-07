from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# file path
train_dir = "./dataset/train_image"
val_dir = "./dataset/val_image"
checkpoint_path_1 = './model/checkpoint_1.h5'
checkpoint_path_2 = './model/checkpoint_2.h5'
checkpoint_path = './model/checkpoint.h5'
inception_json = "./model/inception_model.json"
inception_h5_1 = "./model/inception_h5_1.h5"
inception_h5_2 = "./model/inception_h5_2.h5"
inception_h5_load_from = "./model/inception_h5_load_from.h5"
inception_h5_save_to = "./model/inception_h5_save_to.h5"


def create_model(num_classes):
    # create a inception v3 pre-trained model
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

    # serialize model to json
    model_json = model.to_json()
    with open(inception_json, 'w') as f:
        f.write(model_json)

    return model


def fine_tune(model, nb_train_samples, nb_val_samples, epochs_1=15, epochs_2=30, batch_size=16,
              img_width=299, img_height=299):
    train_data_gen = ImageDataGenerator(preprocessing_function=pre_process,
                                        horizontal_flip=True,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        rotation_range=20)
    val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)

    # data augmentation
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
        callbacks=[ModelCheckpoint(checkpoint_path_1, save_best_only=True, verbose=1, monitor='val_acc', mode='max')])

    model.save_weights(inception_h5_1)

    # freeze only 172 first layer
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # recompile model and train again
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs_2,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        callbacks=[ModelCheckpoint(checkpoint_path_2, save_best_only=True, verbose=1, monitor='val_acc', mode='max')])

    model.save_weights(inception_h5_2)


# def fine_tune_from_saved(nb_train_samples, nb_val_samples, img_width=299, img_height=299,
#                          epochs=100, batch_size=16, nb_freeze=0):
#     # load json and create model
#     with open(inception_json, 'r') as f:
#         loaded_model_json = f.read()
#     loaded_model = model_from_json(loaded_model_json)
#
#     # load weights into new model
#     loaded_model.load_weights(inception_h5_load_from)
#
#     for layer in loaded_model.layers[:nb_freeze]:
#         layer.trainable = False
#     for layer in loaded_model.layers[nb_freeze:]:
#         layer.trainable = True
#
#     # recompile and augmentation
#     loaded_model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy')
#     train_datagen = ImageDataGenerator(
#         preprocessing_function=pre_process,
#         horizontal_flip=True,
#         zoom_range=0.2,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rotation_range=20)
#
#     val_datagen = ImageDataGenerator(preprocessing_function=pre_process)
#
#     train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(img_width, img_height),
#         batch_size=batch_size,
#         class_mode='categorical')
#
#     val_generator = val_datagen.flow_from_directory(
#         val_dir,
#         target_size=(img_width, img_height),
#         batch_size=batch_size,
#         class_mode='categorical')
#     loaded_model.fit_generator(
#         train_generator,
#         steps_per_epoch=nb_train_samples // batch_size,
#         epochs=epochs,
#         validation_data=val_generator,
#         validation_steps=nb_val_samples // batch_size,
#         callbacks=[EarlyStopping(monitor='val_loss'),
#                    ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1, monitor='val_acc', mode='max')])
#     loaded_model.save_weights(inception_h5_save_to)


def pre_process(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def train_model():
    nb_classes = 100
    nb_train_samples = 7963
    nb_val_samples = 300
    print("Creating model...")
    model = create_model(nb_classes)
    print("Training model...")
    fine_tune(model, nb_train_samples, nb_val_samples)
    # fine_tune_from_saved(nb_train_samples, nb_val_samples)


if __name__ == '__main__':
    train_model()
