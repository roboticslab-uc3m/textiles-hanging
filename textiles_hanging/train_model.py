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

from models import HANGnet, HANGnet_dropout, HANGnet_shallow, HANGnet_large

models_with_name = [('HANGnet', HANGnet), ('HANGnet_dropout', HANGnet_dropout), ('HANGnet_shallow', HANGnet_shallow),
                    ('HANGnet_large', HANGnet_large)]

optimizers_with_name = [('adam', Adam), ('sgd', SGD), ('rmsprop', RMSprop)]
batch_sizes = [32, 64, 128]

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
    if optimizer.lower() == 'adam':
        optimizer = Adam()
    elif optimizer.lower() == 'sgd':
        optimizer = SGD()
    elif optimizer.lower() == 'rmsprop':
        optimizer = RMSprop()

    INPUT_SHAPE = (240, 320, 1)  # This seems to be a reminder, as it is not used later

    # Data is loaded here
    with np.load(os.path.abspath(os.path.expanduser(training_data))) as data:
        X = data['X']
        Y = data['Y'][:, 1, :].reshape((-1, 3))
    logging.info('Loaded training examples (X): {}'.format(X.shape))
    logging.info('Loaded labels (Y): {}'.format(Y.shape))

    # Scaling. As the input is a depth map, to apply scaling we need to unravel all the data
    # as if there was only one feature (depth)
    logging.info("Scaling input features (X)...")
    scaler_X = StandardScaler()
    X_shape = X.shape
    X_1 = np.reshape(X, (X_shape[0], X_shape[1]*X_shape[2]))
    X_2 = np.reshape(X_1, (X_1.shape[0]*X_1.shape[1], 1))
    X_3 = np.where(X_2 >= 10, 10, X_2)  # Crop infinity to 10m (for data scaling)
    X_scaled = scaler_X.fit_transform(X_3)
    X_scaled_1 = np.reshape(X_scaled, X_1.shape)
    X_scaled_original_shape = np.reshape(X_scaled_1, X_shape)
    # store these off for predictions with unseen data
    X_means = scaler_X.mean_
    X_stds = scaler_X.scale_
    np.savez('X_scaling.npz', X_means=X_means, X_stds=X_stds) # Not now

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
    X_scaled = X_scaled_original_shape[:, :, :, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2)

    for model_name, model_generator in models_with_name:
        model = model_generator()

        logging.debug(model.summary())
        if do_not_train:
            exit(0)

        for batch_size in batch_sizes:
            for opt_name, opt_generator:
                optimizer = opt_generator()
                full_log_dir = os.path.abspath(os.path.expanduser(os.path.join(log_dir,
                                                                               "{}_{}_{}_{}".format(model_name,
                                                                                                    n_epoch,
                                                                                                    batch_size,
                                                                                                    opt_name))))
                tensorboard = TensorBoard(log_dir=full_log_dir)
                model.compile(loss="mse", optimizer=optimizer, metrics=["mse", "mae"])

                history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch,
                                    verbose=VERBOSE, validation_split=validation_split, callbacks=[tensorboard])
                model.save_weights('hangnet-weights_{}_{}_{}_{}.h5'.format(model_name, n_epoch, batch_size, opt_name))
                with open('hangnet-history_{}_{}_{}_{}.pickle'.format(model_name, n_epoch, batch_size, opt_name),
                          'wb') as f:
                    pickle.dump(history.history, f)

                score = model.evaluate(X_test, y_test, verbose=VERBOSE)
                logging.info("Test score:")
                for metric, s in zip(model.metrics_names, score):
                    logging.info("\t- {}: {}".format(metric, s))

                with open('hangnet-score_{}_{}_{}_{}.pickle'.format(model_name, n_epoch, batch_size, opt_name),
                          'wb') as f:
                    pickle.dump(score, f)

                # Plot loss to a file for feedback
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.savefig("loss_val_loss_{}_{}_{}_{}.png".format(model_name, n_epoch, batch_size, opt_name))
