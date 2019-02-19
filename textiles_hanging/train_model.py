import logging
import os

import begin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam


# define a HANGnet network
def HANGnet(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(180, 240, 1)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(ZeroPadding2D((1, 1)))

    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    #top layer of the VGG net
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='linear'))

    if weights_path:
        model.load_weights(weights_path)

    return model


@begin.start(auto_convert=True)
@begin.logging
def main(training_data: 'npz file containing training data',
         scale_output: 'Scale the output for training'=True,
         n_epoch=20, batch_size=128, optimizer: 'Optimizer to be used [adam, sgd, rmsprop]'='adam',
         validation_split=0.2):

    logging.info("Current configuration: ")
    logging.info("Number of epochs: {}".format(n_epoch))
    logging.info("Batch size: {}".format(batch_size))
    logging.info("Optimizer: {}".format(optimizer))
    if scale_output:
        logging.info("Scaling output for training")
    VERBOSE = 1
    if optimizer.lower() == 'adam':
        optimizer = Adam()
    elif optimizer.lower() == 'sgd':
        optimizer = SGD()
    elif optimizer.lower() == 'rmsprop':
        optimizer = RMSprop()

    INPUT_SHAPE = (240, 320, 1)  # This seems to be a reminder, as it is not used later

    # data: shuffled and split between train and test sets
    with np.load(os.path.abspath(os.path.expanduser(training_data))) as data:
        X = data['X']
        Y = data['Y'][:, 1, :].reshape((-1, 3))
    logging.info('Loaded training examples (X): {}'.format(X.shape))
    logging.info('Loaded labels (Y): {}'.format(Y.shape))

    # Scaling. As the input is a depth map, to apply scaling we need to unravel all the data
    # as if there was only one feature (depth)
    scaler_X = StandardScaler()
    X_shape = X.shape
    X_1 = np.reshape(X, (X_shape[0], X_shape[1]*X_shape[2]))
    X_2 = np.reshape(X_1, (X_1.shape[0]*X_1.shape[1], 1))
    X_scaled = scaler_X.fit_transform(X_2)
    X_scaled_1 = np.reshape(X_scaled, X_1.shape)
    X_scaled_original_shape = np.reshape(X_scaled_1, X_shape)
    # store these off for predictions with unseen data
    X_means = scaler_X.mean_
    X_stds = scaler_X.scale_
    np.savez('X_scaling.npz', X_means=X_means, X_stds=X_stds)

    if scale_output:
        scaler_Y = StandardScaler()
        Y_scaled = scaler_Y.fit_transform(Y)
        # store these off for predictions with unseen data
        Y_means = scaler_Y.mean_
        Y_stds = scaler_Y.scale_
        np.savez('Y_scaling.npz', Y_means=Y_means, Y_stds=Y_stds)
    else:
        Y_scaled = Y

    # Train / test split
    X_scaled = X_scaled_original_shape[:, np.newaxis, :, :]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2)
    K.set_image_dim_ordering("tf")

    # Test pretrained model
    model = HANGnet()

    print(model.summary())
    exit(0)

    model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy", "mse"])

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch,
                        verbose=VERBOSE, validation_split=validation_split)

    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    logging.info("\nTest score:", score[0])
    logging.info('Test accuracy:', score[1])

    # list all data in history
    logging.info(history.history.keys())
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