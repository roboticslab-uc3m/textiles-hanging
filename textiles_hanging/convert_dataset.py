import os
import logging
import csv

import begin
from tqdm import tqdm
import OpenEXR
import Imath
import numpy as np

# Inspiration: https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b


def numpy_from_exr(filepath):
    input = OpenEXR.InputFile(filepath)  # Loop in the final version
    dw = input.header()['dataWindow']
    isize = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    imgStr = input.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    img = np.fromstring(imgStr, dtype=np.float32)
    img.shape = (isize[1], isize[0])  # Numpy arrays are (row, col)
    return img


def visualize_depth_with_background(img):
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    unique, counts = np.unique(img, return_counts=True)
    histogram_ind = np.argsort(unique)
    histogram = unique[histogram_ind]
    logging.debug(histogram)
    logging.debug(histogram[-1])
    logging.debug(histogram[-2])

    plt.imshow(np.where(img == histogram[-1], histogram[-2], img), cmap=plt.cm.RdGy)
    plt.show()


@begin.start(auto_convert=True)
@begin.logging
def main(in_folder: 'Input folder containing the dataset'='.', out_folder: 'Output folder for the npz file'='.',
         scale: 'If True, resizes images to 240x180'=False):
    """
    Converts a dataset generated by blender (EXR images and CSV trajectories) to a numpy npz file containing all
    the data.
    """
    in_folder = os.path.abspath(os.path.expanduser(in_folder))
    out_folder = os.path.abspath(os.path.expanduser(out_folder))
    logging.info("Input folder: {}".format(in_folder))
    logging.info("Output folder: {}".format(out_folder))

    img_prefix, img_ext = 'img-', '.exr'
    exr_files = [f for f in os.listdir(in_folder) if img_prefix in f and img_ext in f]
    csv_files = [f[:f.find('.')]+'.csv' for f in exr_files]
    logging.info("{} files found.".format(len(exr_files)))
    logging.debug(exr_files)
    logging.debug(csv_files)

    # Reserve memory for all the images to be loaded
    X = np.zeros(240, 320, (len(exr_files)))
    Y = np.zeros(2, 3, (len(exr_files)))

    # Load images
    for i, (exr_file, csv_file) in tqdm(enumerate(zip(exr_files, csv_files))):
        # Load exr file
        X[i, :, :] = numpy_from_exr(os.path.join(in_folder, exr_file))

        # Load csv file
        reader = csv.reader(open(os.path.join(in_folder, csv_file), "r"), delimiter=" ")
        trajectory_data = list(reader)
        trajectory = np.array(trajectory_data).astype("float")
        Y[0, :, i] = trajectory[4]
        Y[1, :, i] = trajectory[-1]

    # Save files
    np.savez_compressed(os.path.join(out_folder, 'data.npz'), X=X, Y=Y)




