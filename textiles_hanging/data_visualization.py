import os
import logging

import begin
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def visualize_depth_with_background(img):
    unique, counts = np.unique(img, return_counts=True)
    histogram_ind = np.argsort(unique)
    histogram = unique[histogram_ind]
    logging.debug(histogram)
    logging.debug(histogram[-1])
    logging.debug(histogram[-2])

    plt.imshow(np.where(img == histogram[-1], histogram[-2], img), cmap=plt.cm.RdGy)
    plt.show()


def visualize_depth_with_truncated_background(img, show=True, title=None):
    img_truncated = np.where(img >= 10, 10, img)  # Crop infinity to 10m (for data scaling)
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
    :param data: Vector of 3D trajectories [dims -> (n, m, 3), n: n samples, m: traj length]
    :param show: Shows the figure if True
    """
    logging.debug(data.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for traj in data:
        color = 'b'
        if traj[-1, 2] < 0.12:
            color = 'r'
            continue
        elif traj[-1, 2] > 0.81:
            color = 'g'
            continue
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-{}'.format(color), linewidth=1)
    if show:
        plt.show()


@begin.start(auto_convert=True)
@begin.logging
def main(training_data: 'npz file containing training data',
         view_inputs: 'show 5 random input examples'=False,
         full_trajs: 'npz file containing full trajectories'=None):
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
        for i in range(5):
            random_index = np.random.randint(0, X.shape[0])
            visualize_depth_with_truncated_background(X[random_index, :, :], show=False, title=random_index)

    if full_trajs:
        plt.figure('Trajectories')
        for traj in Y_full:
            color = 'b'
            if traj[-1, 2] < 0.12:
                color='r'
            elif traj[-1, 2] > 0.81:
                color='g'
            plt.plot(range(traj.shape[0]), traj[:, 2], '-{}'.format(color))

        visualize_3D_trajectories(Y_full, show=False)
    plt.show()
