import logging
import os
import pickle
from time import time

import begin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import TensorBoard

from models import HANGnet, HANGnet_dropout, HANGnet_shallow, HANGnet_large, HANGnet_very_large
from generators import HangingDataGenerator
from convert_dataset import get_dataset_filenames

models_with_name = [('HANGnet', HANGnet)] #,
#                    ('HANGnet_dropout', HANGnet_dropout)]#, ('HANGnet_shallow', HANGnet_shallow),
#                    ('HANGnet_large', HANGnet_large), ('HANGnet_very_large', HANGnet_very_large)]

optimizers_with_name = [('adam', Adam)]  #, ('sgd', SGD), ('rmsprop', RMSprop)]
batch_sizes = [32]#, 64, 128]


def custom_loss(y_true, y_pred):
    squared_components = K.square(y_true-y_pred)
    return 0.2*squared_components[0]+0.2*squared_components[1]+0.6*squared_components[2]


@begin.start(auto_convert=True)
@begin.logging
def main(training_data_dir: 'folder containing training data',
         log_dir: 'TensorBoard logs go here'='./logs',
         results_dir: 'Weights, train-validation partition and other results go here'='./results',
         do_not_train: 'If present, exits after summary'=False,
         n_epoch=20, batch_size=128, optimizer: 'Optimizer to be used [adam, sgd, rmsprop]'='adam',
         validation_split=0.2):

    mpl.use('Agg')  # Plot without X

    logging.info("Current configuration: ")
    logging.info("Number of epochs: {}".format(n_epoch))
    logging.info("Batch size: {}".format(batch_size))
    logging.info("Optimizer: {}".format(optimizer))
    VERBOSE = 1
    if optimizer.lower() == 'adam':
        optimizer = Adam()
    elif optimizer.lower() == 'sgd':
        optimizer = SGD()
    elif optimizer.lower() == 'rmsprop':
        optimizer = RMSprop()

    INPUT_SHAPE = (240, 320, 1)  # This seems to be a reminder, as it is not used later

    # Get ids of files to load
    if os.path.exists(os.path.join(training_data_dir, 'dataset_files.pickle')):
        with open(os.path.join(training_data_dir, 'dataset_files.pickle'), 'rb') as f:
            dataset_files = pickle.load(f)
    else:
        dataset_files = get_dataset_filenames(training_data_dir)
        with open('hangnet-last-dataset-filenames.pickle', 'wb') as f:
            pickle.dump(dataset_files, f)

    # Split dataset
    np.random.shuffle(dataset_files)
    test_size = int(np.floor(0.2*len(dataset_files)))
    validation_size = int(np.floor(validation_split*(1-0.2)*len(dataset_files)))
    test_files = dataset_files[:test_size]
    validation_files = dataset_files[test_size:test_size+validation_size]
    train_files = dataset_files[test_size+validation_size:]
    logging.info("Training set: {} examples".format(len(train_files)))
    logging.info("Validation set: {} examples".format(len(validation_files)))
    logging.info("Test set: {} examples".format(len(test_files)))
    t = time()
    with open(os.path.join(results_dir, 'hangnet-test-files-{}.pickle'.format(t)), 'wb') as f:
        pickle.dump(test_files, f)
    with open(os.path.join(results_dir, 'hangnet-validation-files-{}.pickle'.format(t)), 'wb') as f:
        pickle.dump(validation_files, f)
    with open(os.path.join(results_dir, 'hangnet-train-files-{}.pickle'.format(t)), 'wb') as f:
        pickle.dump(train_files, f)

    # Create data generator
    params = {'batch_size': batch_size, 'resize': True,  'shuffle': True}

    training_generator = HangingDataGenerator(train_files, training_data_dir, **params)
    validation_generator = HangingDataGenerator(validation_files, training_data_dir, **params)
    test_generator = HangingDataGenerator(test_files, training_data_dir, **params)

    for model_name, model_generator in models_with_name:
        model = model_generator()

        logging.info("Training model: {}".format(model_name))
        logging.debug(model.summary())
        if do_not_train:
            exit(0)

        for batch_size in batch_sizes:
            for opt_name, opt_generator in optimizers_with_name:
                # Generate paths for logging to files (results/logs)
                full_log_dir = os.path.abspath(os.path.expanduser(os.path.join(log_dir,
                                                                               "{}_{}_{}_{}".format(model_name,
                                                                                                    n_epoch,
                                                                                                    batch_size,
                                                                                                    opt_name))))
                full_result_dir = os.path.join(results_dir, model_name)
                if not os.path.exists(full_result_dir):
                    os.makedirs(full_result_dir)
                weights_path = os.path.join(full_result_dir, 'hangnet-weights_{}_{}_{}_{}.h5'.format(model_name,
                                                                                                     n_epoch,
                                                                                                     batch_size,
                                                                                                     opt_name))
                history_path = os.path.join(full_result_dir, 'hangnet-history_{}_{}_{}_{}.pickle'.format(model_name,
                                                                                                         n_epoch,
                                                                                                         batch_size,
                                                                                                         opt_name))
                score_path = os.path.join(full_result_dir, 'hangnet-score_{}_{}_{}_{}.pickle'.format(model_name,
                                                                                                     n_epoch,
                                                                                                     batch_size,
                                                                                                     opt_name))
                figure_path = os.path.join(full_result_dir, "loss_plot_{}_{}_{}_{}.png".format(model_name, n_epoch,
                                                                                               batch_size, opt_name))

                # Create a tensorboard to log stats
                tensorboard = TensorBoard(log_dir=full_log_dir, write_graph=False)

                # Train model
                optimizer = opt_generator(lr=0.0001)
                model.compile(loss=custom_loss, optimizer=optimizer, metrics=["mse"])

                history = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                                              use_multiprocessing=True, workers=6,
                                              epochs=n_epoch,
                                              verbose=VERBOSE, callbacks=[tensorboard])

                # Evaluate and save results to files
                model.save_weights(weights_path)
                with open(history_path, 'wb') as f:
                    pickle.dump(history.history, f)

                score = model.evaluate_generator(generator=test_generator, verbose=VERBOSE,
                                                 use_multiprocessing=True, workers=6,)
                logging.info("Test score:")
                for metric, s in zip(model.metrics_names, score):
                    logging.info("\t- {}: {}".format(metric, s))

                with open(score_path, 'wb') as f:
                    pickle.dump(score, f)

                # Plot loss to a file for feedback
                plt.figure()
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.savefig(figure_path)
