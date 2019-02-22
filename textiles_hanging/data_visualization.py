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


@begin.start(auto_convert=True)
@begin.logging
def main(training_data: 'npz file containing training data'):
    logging.info("Using dataset: {}".format(training_data))

    # Data is loaded here
    with np.load(os.path.abspath(os.path.expanduser(training_data))) as data:
        X = data['X']
        Y = data['Y'][:, 1, :].reshape((-1, 3))
    logging.info('Loaded training examples (X): {}'.format(X.shape))
    logging.info('Loaded labels (Y): {}'.format(Y.shape))

    logging.info("Analysis:")
    # Compute classes:
    unique, counts = np.unique(Y[:, 2] > 0.2, return_counts=True)
    n_hanged = counts[np.argsort(unique)[1]]
    logging.info("\t- Hanged: {}/{} - {}%".format(n_hanged, Y.shape[0], n_hanged/Y.shape[0]*100))

    visualize_3D_scatterplot(Y)
