import logging
import os
import pickle

import begin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.applications import VGG16, vgg16


def HANGnet_VGG(weights_path=None):
    vgg =  VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg.layers[:-4]:
        layer.trainable = False

    model = Sequential()
    model.add(vgg)

    # Add new layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='linear'))

    if weights_path:
        model.load_weights(weights_path)

    return model


@begin.start(auto_convert=True)
@begin.logging
def main(training_data: 'npz file containing training data',
         log_dir: 'TensorBoard logs go here'='./logs',
         do_not_train: 'If present, exits after summary'=False,
         scale_output: 'Scale the output for training'=True,
         n_epoch=20, batch_size=128, optimizer: 'Optimizer to be used [adam, sgd, rmsprop]'='adam',
         validation_split=0.2):

    mpl.use('Agg')  # Plot without X

    logging.info("Current configuration: ")
    logging.info("Number of epochs: {}".format(n_epoch))
    logging.info("Batch size: {}".format(batch_size))
    logging.info("Optimizer: {}".format(optimizer))
    if scale_output:
        logging.info("Scaling output for training")
    VERBOSE = 1
    opt_name = optimizer.lower()
    if optimizer.lower() == 'adam':
        optimizer = Adam()
    elif optimizer.lower() == 'sgd':
        optimizer = SGD()
    elif optimizer.lower() == 'rmsprop':
        optimizer = RMSprop()

    INPUT_SHAPE = (224, 224, 3)  # This seems to be a reminder, as it is not used later

    # Data is loaded here
    with np.load(os.path.abspath(os.path.expanduser(training_data))) as data:
        X = data['X']  # (N, 224, 224, 3)
        Y = data['Y'][:, 1, :].reshape((-1, 3))
    logging.info('Loaded training examples (X): {}'.format(X.shape))
    logging.info('Loaded labels (Y): {}'.format(Y.shape))

    # Scaling. As the input is a depth map, to apply scaling we need to unravel all the data
    # as if there was only one feature (depth)
    logging.info("Scaling input features (X)...")
    X_scaled = vgg16.preprocess_input(X.copy())

    if scale_output:
        logging.info("Scaling output features (Y)...")
        scaler_Y = StandardScaler()
        Y_scaled = scaler_Y.fit_transform(Y)
        # store these off for predictions with unseen data
        Y_means = scaler_Y.mean_
        Y_stds = scaler_Y.scale_
        np.savez('Y_scaling.npz', Y_means=Y_means, Y_stds=Y_stds)
    else:
        Y_scaled = Y

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2)

    model = HANGnet_VGG()

    logging.debug(model.summary())
    if do_not_train:
        exit(0)

    full_log_dir = os.path.abspath(os.path.expanduser(os.path.join(log_dir, "{}_{}_{}_{}".format("HANGnet_VGG",
                                                                                                 n_epoch,
                                                                                                 batch_size,
                                                                                                 opt_name))))
    tensorboard = TensorBoard(log_dir=full_log_dir, write_graph=False, histogram_freq=5, batch_size=batch_size,
                              write_grads=True)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mse", "mae"])

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=VERBOSE,
                        validation_split=validation_split, callbacks=[tensorboard])
    model.save_weights('hangnet-weights_{}_{}_{}_{}.h5'.format("HANGnet_VGG", n_epoch, batch_size, opt_name))
    with open('hangnet-history_{}_{}_{}_{}.pickle'.format("HANGnet_VGG", n_epoch, batch_size, opt_name),
              'wb') as f:
        pickle.dump(history.history, f)

    score = model.evaluate(X_test, y_test, verbose=VERBOSE)
    logging.info("Test score:")
    for metric, s in zip(model.metrics_names, score):
        logging.info("\t- {}: {}".format(metric, s))

    with open('hangnet-score_{}_{}_{}_{}.pickle'.format("HANGnet_VGG", n_epoch, batch_size, opt_name),
              'wb') as f:
        pickle.dump(score, f)

    # Plot loss to a file for feedback
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss_val_loss_{}_{}_{}_{}.png".format("HANGnet_VGG", n_epoch, batch_size, opt_name))
