import os
import logging

import begin
from tqdm import tqdm
import numpy as np
from skimage.transform import resize as sk_resize


@begin.start(auto_convert=True)
@begin.logging
def main(in_file: 'Input npz file containing the dataset'='.', out_folder: 'Output folder for the npz file'='.',
         imagenet_size: 'IF True, resizes to 224x224 (and crops'=False):
    """
    Rescales an existing dataset in npz format
    """
    logging.info("Input file: {}".format(in_file))
    logging.info("Output folder: {}".format(out_folder))
    if imagenet_size:
        logging.info("Using imagenet size (224, 224, 3)")

    with np.load(os.path.abspath(os.path.expanduser(in_file))) as data:
        X_old = data['X']
        Y = data['Y']

    if imagenet_size:
        X = np.zeros((X_old.shape[0], 224, 224, 3))
    else:
        X = np.zeros((X_old.shape[0], 224, 224))

    # Resize images
    for i in tqdm(range(X_old.shape[0]), total=X_old.shape[0]):
        if imagenet_size:
            X[i, :, :, 0] = sk_resize(X_old[i, :, :], (224, 299), anti_aliasing=True)[:, 37:261]
            X[i, :, :, 1] = X[i, :, :, 0]
            X[i, :, :, 2] = X[i, :, :, 0]
        else:
            X[i, :, :] = sk_resize(X_old[i, :, :], (180, 240), anti_aliasing=True)

    # Save files
    np.savez_compressed(os.path.join(out_folder, 'data-resized{}.npz'.format('-imagenet' if imagenet_size else '')),
                        X=X, Y=Y)
