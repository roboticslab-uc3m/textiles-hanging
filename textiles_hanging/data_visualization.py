import os
import logging

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
from sklearn.preprocessing import StandardScaler


logging.getLogger("matplotlib").setLevel(logging.WARNING)


def visualize_depth_with_background(img, show=True):
    """
    Visualize a depth image that has a nan as background, by substituting that value with the previous one in the
    histogram. If the image background is composed of several close values, use visualize_depth_with_thresholded_background()
    instead.
    :param img: Image to visualize
    :param: show: Shows the figure if True
    """
    unique, counts = np.unique(img, return_counts=True)
    histogram_ind = np.argsort(unique)
    histogram = unique[histogram_ind]
    logging.debug(histogram)
    logging.debug(histogram[-1])
    logging.debug(histogram[-2])

    plt.imshow(np.where(img == histogram[-1], histogram[-2], img), cmap=plt.cm.RdGy)
    if show:
        plt.show()


def visualize_depth_with_thresholded_background(img, threshold=10, show=True, title=None):
    """
    Visualize a depth image that has several close values as background, by substituting values larger than the
    threshold by the threshold value.
    :param img: Image to visualize
    :param threshold: Threshold to apply
    :param show: Shows the figure if True
    :param title: Title on the visualization window
    """
    img_truncated = np.where(img >= threshold, threshold, img)  # Crop infinity to 10m (for data scaling)
    plt.figure()
    plt.imshow(img_truncated, cmap=plt.cm.RdGy)
    if title:
        plt.title(title)
    if show:
        plt.show()


def visualize_3D_scatterplot(data, show=True):
    """
    Shows a 3D scatterplot of the input data
    :param data: Vector of 3D points [dims->(n, 3)]
    :param: show: Shows the figure if True
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0][:], data[:, 1][:], data[:, 2][:], s=1, c='r', marker='o')
    if show:
        plt.show()


def visualize_3D_trajectories(data, show=True):
    """
    Shows the different trajectories of the input data
    :param data: Vector of 3D trajectories [dims -> (n, m, 3), n: n samples, m: trajectory length]
    :param show: Shows the figure if True
    """
    logging.debug(data.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for traj in data:
        color = 'b'  # Blue points are points on the floor
        if traj[-1, 2] < 0.12:
            color = 'r'
            continue  # Remove/comment this to show red points (still in movement)
        elif traj[-1, 2] > 0.81:
            color = 'g'
            continue  # Remove/comment this to show green points (hanged)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-{}'.format(color), linewidth=1)
    if show:
        plt.show()


def visualize_input_histogram(data, show=True):
    logging.debug(data.shape)
    fig = plt.figure()
    plt.hist(data[data <= 2].ravel(), bins=50)
    if show:
        plt.show()


@begin.start(auto_convert=True)
@begin.logging
def main(training_data: 'npz file containing training data',
         view_inputs: 'show 5 random input examples'=False,
         full_trajs: 'npz file containing full trajectories'=None,
         view_results: 'Show a visualization of the predicted vs real'=False,
         model_name: 'Name of the model to use for prediction. Supported: []'.format(list(models_with_name.keys()))=None,
         model_weights: 'File with the weights to load for prediction'=None,
         x_scaling: 'File with the scaling used for X in training'=None,
         y_scaling: 'File with the scaling used for Y in training'=None):
    """
    Visualize different aspects from a given dataset. Each flag enables/disables the corresponding visualization.
    """
    logging.info("Using dataset: {}".format(training_data))

    # Data is loaded here
    with np.load(os.path.abspath(os.path.expanduser(training_data))) as data:
        X = data['X']
        Y = data['Y'][:, 1, :].reshape((-1, 3))
    logging.info('Loaded training examples (X): {}'.format(X.shape))
    logging.info('Loaded labels (Y): {}'.format(Y.shape))

    if full_trajs:
        with np.load(os.path.abspath(os.path.expanduser(full_trajs))) as data:
            Y_full = data['Y']
        logging.info('Loaded full trajectories (Y_full): {}'.format(Y_full.shape))

    logging.info("Analysis:")
    # Compute classes:
    unique, counts = np.unique(Y[:, 2] > 0.2, return_counts=True)
    n_hanged = counts[np.argsort(unique)[1]]
    logging.info("\t- Hanged (naive): {}/{} - {}%".format(n_hanged, Y.shape[0], n_hanged/Y.shape[0]*100))

    unique, counts = np.unique(Y[:, 2] < 0.12, return_counts=True)
    n_floor = counts[np.argsort(unique)[1]]
    logging.info("\t- Hanged (floor): {}/{} - {}%".format(n_floor, Y.shape[0], n_floor / Y.shape[0] * 100))
    unique, counts = np.unique((Y[:, 2] >= 0.12) & (Y[:, 2] < 0.81), return_counts=True)
    n_midair = counts[np.argsort(unique)[1]]
    logging.info("\t- Hanged (midair): {}/{} - {}%".format(n_midair, Y.shape[0], n_midair / Y.shape[0] * 100))
    unique, counts = np.unique(Y[:, 2] >= 0.81, return_counts=True)
    n_true_hanged = counts[np.argsort(unique)[1]]
    logging.info("\t- Hanged (true): {}/{} - {}%".format(n_true_hanged, Y.shape[0], n_true_hanged / Y.shape[0] * 100))

    visualize_3D_scatterplot(Y, show=False)

    if view_inputs:
        logging.info("Showing input samples...")
        for i in range(5):
            random_index = np.random.randint(0, X.shape[0])
            visualize_depth_with_thresholded_background(X[random_index, :, :], show=False, title=random_index)
            visualize_input_histogram(X[random_index, :, :], show=False)

    if full_trajs:
        logging.info("Showing input trajectories...")
        plt.figure('Trajectories')
        for traj in Y_full:
            color = 'b'
            if traj[-1, 2] < 0.12:
                color='r'
            elif traj[-1, 2] > 0.81:
                color='g'
            plt.plot(range(traj.shape[0]), traj[:, 2], '-{}'.format(color))

        visualize_3D_trajectories(Y_full, show=False)

    if view_results:
        logging.info("Showing results...")
        try:
            model = models_with_name[model_name](os.path.abspath(os.path.expanduser(model_weights)))
        except NameError:
            logging.error("Model cannot be loaded. Missing model? Missing Tensorflow/Keras?")
            exit(2)

        # Select data
        indices = np.random.choice(range(X.shape[0]), 500, replace=False)
        X_test = np.take(X, indices, axis=0)
        X_test = X_test[:, :, :, np.newaxis]
        Y_test = np.take(Y, indices, axis=0)

        if x_scaling and y_scaling:
            # Convert data to normalized values (or use normalized data instead)
            # To do this, you'll need the normalization matrices
            with np.load(os.path.abspath(os.path.expanduser(x_scaling))) as data:
                X_means = data['X_means']
                X_stds = data['X_stds']
            X_test = (X_test-X_means)/X_stds

            with np.load(os.path.abspath(os.path.expanduser(y_scaling))) as data:
                Y_means = data['Y_means']
                Y_stds = data['Y_stds']
            # Do not scale Y, we need to revert the scaling of the predicted output instead

        # New scaling
        thres = 2
        X_test = np.where(X_test >= thres, thres, X_test)
        X_test = 2*X_test/thres-1

        # Predict and revert output to original scale
        Y_pred = model.predict(X_test)
        #Y_pred = (Y_pred*Y_stds)+Y_means

        # Plot predictions with actual data
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(Y_test[:, 0], Y_test[:, 1], Y_test[:, 2], s=1, c='g', marker='o', label='Ground Truth')
        ax.scatter(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], s=1, c='b', marker='o', label='Predicted')
        #for start, end in zip(Y_test, Y_pred):
        #    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-', label='Error')

    plt.show()
