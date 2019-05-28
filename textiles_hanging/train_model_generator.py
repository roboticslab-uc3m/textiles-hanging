import logging
import os
import pickle
from time import time
import csv

import begin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import TensorBoard
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from models import *
from generators import HangingDataGenerator, HangingBinaryDataGenerator
from convert_dataset import get_dataset_filenames

models_with_name = [('HANGnet_classify_regularized', HANGnet_classify_regularized)]
#                    ('HANGnet_classify', HANGnet_classify)]
#                    ('HANGnet', HANGnet),
#                    ('HANGnet_dropout', HANGnet_dropout)]#, ('HANGnet_shallow', HANGnet_shallow),
#                    ('HANGnet_large', HANGnet_large), ('HANGnet_very_large', HANGnet_very_large)]

optimizers_with_name = [('adam', Adam)]  #, ('sgd', SGD), ('rmsprop', RMSprop)]
batch_sizes = [32]#, 64, 128]
learning_rates = [0.0001]  #, 0.0002, 0.0004, 0.001]
regularization_strengths = [0.001]  #, 0.002, 0.043, 0.01]


def custom_loss(y_true, y_pred):
    squared_components = K.square(y_true-y_pred)
    return 0.033 * squared_components[:, 0] + 0.033 * squared_components[:, 1] + 0.33 * squared_components[:, 2]


@begin.start(auto_convert=True)
@begin.logging
def main(training_data_dir: 'folder containing training data',
         log_dir: 'TensorBoard logs go here'='./logs',
         results_dir: 'Weights, train-validation partition and other results go here'='./results',
         do_not_train: 'If present, exits after summary'=False,
         n_epoch=20, batch_size=128, optimizer: 'Optimizer to be used [adam, sgd, rmsprop]'='adam',
         validation_split=0.2):

    mpl.use('Agg')  # Plot without X
    training_data_dir = os.path.abspath(os.path.expanduser(training_data_dir))

    logging.info("Current configuration: ")
    logging.info("Training data folder: {}".format(training_data_dir))
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

    # Get labels to make a stratified split
    # Note: be sure to use the same criteria for the labels as the generator used for training
    dataset_files_as_array = np.array(dataset_files)
    dataset_labels = np.empty(dataset_files_as_array.shape, dtype=np.uint8)
    for i, row in tqdm(enumerate(dataset_files_as_array)):
        reader = csv.reader(open(os.path.join(training_data_dir, row + '.csv'), "r"), delimiter=" ")
        trajectory_data = np.array(list(reader)).astype("float")
        dataset_labels[i] = trajectory_data[-1, 2] < 0.81
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)  # Fixed for comparison of models
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    X_indexes, X_test_indexes = sss.split(dataset_files_as_array, dataset_labels).__next__()
    test_files = dataset_files_as_array[X_test_indexes]
    X_train_indexes, X_val_indexes = sss2.split(dataset_files_as_array[X_indexes], dataset_labels[X_indexes]).__next__()
    train_files = dataset_files_as_array[X_indexes][X_train_indexes]
    validation_files = dataset_files_as_array[X_indexes][X_val_indexes]

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

    training_generator = HangingBinaryDataGenerator(train_files, training_data_dir, **params)
    validation_generator = HangingBinaryDataGenerator(validation_files, training_data_dir, **params)
    test_generator = HangingBinaryDataGenerator(test_files, training_data_dir, **params)

    for learning_rate in learning_rates:
        lr_results_dir = os.path.join(results_dir, "lr{}".format(learning_rate)) \
            if len(learning_rates) > 1 else results_dir
        lr_log_dir = os.path.join(log_dir, "lr{}".format(learning_rate)) if len(learning_rates) > 1 else log_dir

        for regularization_strength in regularization_strengths:
            reg_results_dir = os.path.join(lr_results_dir, "reg{}".format(regularization_strength)) \
                if len(regularization_strengths) > 1 else lr_results_dir
            reg_log_dir = os.path.join(log_dir, "reg{}".format(regularization_strength)) \
                if len(regularization_strengths) > 1 else lr_log_dir

            for model_name, model_generator in models_with_name:
                model = model_generator()

                logging.info("Training model: {}".format(model_name))
                logging.debug(model.summary())
                if do_not_train:
                    exit(0)

                for batch_size in batch_sizes:
                    for opt_name, opt_generator in optimizers_with_name:
                        # Generate paths for logging to files (results/logs)
                        full_log_dir = os.path.abspath(os.path.expanduser(os.path.join(reg_log_dir,
                                                                                       "{}_{}_{}_{}".format(model_name,
                                                                                                            n_epoch,
                                                                                                            batch_size,
                                                                                                            opt_name))))
                        full_result_dir = os.path.join(reg_results_dir, model_name)
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
                        figure2_path = os.path.join(full_result_dir, "acc_plot_{}_{}_{}_{}.png".format(model_name, n_epoch,
                                                                                                       batch_size, opt_name))

                        # Create a tensorboard to log stats
                        tensorboard = TensorBoard(log_dir=full_log_dir, write_graph=False)

                        # Train model
                        optimizer = opt_generator(lr=0.0001)
                        class_weights = compute_class_weight('balanced',
                                                             np.unique(dataset_labels[X_indexes][X_train_indexes]),
                                                             dataset_labels[X_indexes][X_train_indexes])
                        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

                        history = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                                                      use_multiprocessing=True, workers=6,
                                                      epochs=n_epoch,
                                                      class_weights=class_weights,
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

                        plt.figure()
                        plt.plot(history.history['acc'])
                        plt.plot(history.history['val_acc'])
                        plt.title('model accuracy')
                        plt.ylabel('acc')
                        plt.xlabel('epoch')
                        plt.legend(['train', 'test'], loc='upper left')
                        plt.savefig(figure2_path)
