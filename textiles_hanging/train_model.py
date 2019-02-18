from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam
import cv2, numpy as np


# define a VGG16 network
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(1,320,240)))  # 3, 224, 224  # Don't know yet if this should be squared
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())

    #top layer of the VGG net
    model.add(Dense(4096, activation='relu'))  # Should I remove these layers too?
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='linear'))

    if weights_path:
        model.load_weights(weights_path)

    return model


if __name__ == "__main__":
    # network and training
    NB_EPOCH = 20
    BATCH_SIZE = 128
    VERBOSE = 1
    OPTIMIZER = Adam()
    VALIDATION_SPLIT = 0.2

    IMG_ROWS, IMG_COLS = 28, 28  # input image dimensions
    NB_CLASSES = 10  # number of outputs = number of digits
    INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)

    # data: shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # replace by our dataset
    K.set_image_dim_ordering("th")

    # Test pretrained model
    model = VGG_16()

    # ToDo: Add training code
    model.compile(loss="mse", optimizer=OPTIMIZER, metrics=["accuracy", "mse"])

    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                        verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    print("\nTest score:", score[0])
    print('Test accuracy:', score[1])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
