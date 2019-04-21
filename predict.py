from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

test_dir = "./dataset/val_image/"
inception_json = "./model/inception_model.json"
checkpoint_path_2 = './model/checkpoint_2.h5'


def evaluate():
    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_data_gen.flow_from_directory(
        test_dir,
        target_size=(299, 299),
        batch_size=1,
        class_mode="categorical"
    )
    file_names = test_generator.filenames
    nb_samples = len(file_names)

    json_file = open(inception_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(checkpoint_path_2)

    # evaluate loaded model on test data
    sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)
    loaded_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    score = loaded_model.predict_generator(test_generator, steps=nb_samples)

    for i in range(0, nb_samples):
        idx = np.argmax(score[i])
        print(idx, score[i][idx])

    y_pred = np.argmax(score, axis=1)
    cm = confusion_matrix(test_generator.classes, y_pred)
    plot_confusion_matrix(cm, idx, normalize=True, title="Normalize confusion matrix")
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalaligment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")


if __name__ == '__main__':
    evaluate()
