import os
import logging
import pickle
from time import time

import begin
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from models import HANGnet, HANGnet_dropout, HANGnet_shallow, HANGnet_large, HANGnet_very_large

    models_with_name = {'HANGnet': HANGnet, 'HANGnet_dropout': HANGnet_dropout,
                        'HANGnet_shallow': HANGnet_shallow,
                        'HANGnet_large': HANGnet_large, 'HANGnet_very_large': HANGnet_very_large}
except ImportError as e:
    models_with_name = {}
    logging.error("Could not load models. Result prediction will be disabled")
    logging.error("Reason: {}".format(str(e)))

from data_visualization import visualize_3D_scatterplot, visualize_3D_trajectories, visualize_input_histogram
from generators import HangingDataGenerator


logging.getLogger("matplotlib").setLevel(logging.WARNING)


@begin.start(auto_convert=True)
@begin.logging
def main(training_data_dir: 'folder containing training data',
         test_set: 'Name of the pickle file containing test set files',
         time_step: 'Instant of the trajectory used for training'=-1,
         model_name: 'Name of the model to use for prediction. Supported:{}'.format(list(models_with_name.keys()))=None,
         model_weights: 'File with the weights to load for prediction'=None):
    """
    Visualize different aspects from a given dataset. Each flag enables/disables the corresponding visualization.
    """
    training_data_dir = os.path.abspath(os.path.expanduser(training_data_dir))
    test_set = os.path.abspath(os.path.expanduser(test_set))
    logging.info("Using dataset: {}".format(training_data_dir))
    logging.info("Using test set: {}".format(test_set))

    # Data is loaded here via generator
    with open(test_set, 'rb') as f:
        test_files = pickle.load(f)

    params = {'batch_size': len(test_files), 'resize': True,  'shuffle': True, 'full_trajectory': True}
    test_generator = HangingDataGenerator(test_files, training_data_dir, **params)
    X, Y = test_generator[0]
    logging.info('Generator size: {}'.format(len(test_generator)))
    logging.info('Loaded training examples (X): {}'.format(X.shape))
    logging.info('Loaded labels (Y): {}'.format(Y.shape))
    logging.info('Using instant {}'.format(time_step))
    Y = Y[:, time_step, :].reshape((-1, 3))
    logging.info('New labels (Y): {}'.format(Y.shape))

    logging.info("Analysis:")
    # Compute classes:    ==== REFACTOR THIS TO USE IN NOTEBOOK ? =================
    try:
        unique, counts = np.unique(Y[:, -1] > 0.2, return_counts=True)
        n_hanged = counts[np.argsort(unique)[1]]
        logging.info("\t- Hanged (naive): {}/{} - {}%".format(n_hanged, Y.shape[0], n_hanged/Y.shape[0]*100))

        unique, counts = np.unique(Y[:, -1] < 0.12, return_counts=True)
        n_floor = counts[np.argsort(unique)[1]]
        logging.info("\t- Hanged (floor): {}/{} - {}%".format(n_floor, Y.shape[0], n_floor / Y.shape[0] * 100))
        unique, counts = np.unique((Y[:, -1] >= 0.12) & (Y[:, 2] < 0.81), return_counts=True)
        n_midair = counts[np.argsort(unique)[1]]
        logging.info("\t- Hanged (midair): {}/{} - {}%".format(n_midair, Y.shape[0], n_midair / Y.shape[0] * 100))
        unique, counts = np.unique(Y[:, 2] >= 0.81, return_counts=True)
        n_true_hanged = counts[np.argsort(unique)[1]]
        logging.info("\t- Hanged (true): {}/{} - {}%".format(n_true_hanged, Y.shape[0], n_true_hanged / Y.shape[0] * 100))
    except IndexError:
        logging.error("\t - Error computing stats for this time step / dataset")

    visualize_3D_scatterplot(Y, show=False)

    logging.info("Computing prediction...")
    try:
        model = models_with_name[model_name](os.path.abspath(os.path.expanduser(model_weights)))
    except NameError:
        logging.error("Model cannot be loaded. Missing model? Missing Tensorflow/Keras?")
        exit(2)

    # Predict and revert output to original scale # ==== LET'S ADAPRT GENERATOR TO KEEP THIS
    Y_pred = model.predict(X)

    logging.info("Showing results...")
    # Plot predictions with actual data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=1, c='g', marker='o', label='Ground Truth')
    ax.scatter(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], s=1, c='b', marker='o', label='Predicted')
    for start, end in zip(Y, Y_pred):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-', label='Error')

    X = Y[:, 0]
    Y_ = Y[:, 1]
    Z = Y[:, 2]
    max_range = np.array([X.max() - X.min(), Y_.max() - Y_.min(), Z.max() - Z.min()]).max() / 2.0

    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y_.max() + Y_.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

    np.savez('results_step{}_{}.npz'.format(time_step, time()), Y=Y, Y_pred=Y_pred)
