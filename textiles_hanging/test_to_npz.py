import os
import pickle

import begin
import numpy as np

from generators import HangingBinaryDataGenerator


@begin.start(auto_convert=True)
def main(test_files: 'Pickle file containing test set filenames',
         input_folder: 'Folder containing the actual files',
         resize: 'Resize the input files (or not)'=True):

    test_files = os.path.abspath(os.path.expanduser(test_files))
    input_folder = os.path.abspath(os.path.expanduser(input_folder))

    # Create data generator
    with open(test_files, 'rb') as f:
        test_files = pickle.load(f)
    params = {'batch_size': 100, 'resize': resize, 'shuffle': False}
    if not resize:
        params['dims']=(240, 320)
    training_generator = iter(HangingBinaryDataGenerator(test_files, input_folder, **params))

    X1, y1 = training_generator.__next__()
    X2, y2 = training_generator.__next__()

    X = np.concatenate((X1, X2), axis=0)
    np.savez("data.npz", X=X)
