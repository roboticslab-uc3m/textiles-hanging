import os
import logging
import csv

import numpy as np
from keras.utils import Sequence
try:
    from skimage.transform import resize as sk_resize
except ImportError:
    logging.error("Scikit-image cannot be found. Please install it (pip install scikit-image)")
    exit(1)

from convert_dataset import numpy_from_exr

exr_filext = '.exr0200.exr'

class HangingDataGenerator(Sequence):
    def __init__(self, data_file_ids, data_folder='.', batch_size=32, dims=(180, 240), resize=True,
                 threshold=True, feature_scaling=True, shuffle=True, full_trajectory=False):
        self.data_file_ids = data_file_ids
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.dims = dims
        self.resize = resize
        self.threshold = threshold
        self.feature_scaling = feature_scaling
        self.shuffle = shuffle
        self.full_trajectory = full_trajectory
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.data_file_ids)

    def _data_generation(self, files_to_load):
        X = np.empty((self.batch_size, self.dims[0], self.dims[1], 1))
        if not self.full_trajectory:
            y = np.empty((self.batch_size, 3))  # 3 -> x, y, z
        else:
            y = np.empty((self.batch_size, 51, 3))  # 3->(x, y, z) 51->points in trajectory (might not work on old data)

        for i, file in enumerate(files_to_load):
            if self.resize:
                X[i, :, :, 0] = sk_resize(numpy_from_exr(os.path.join(self.data_folder, file+exr_filext)), self.dims,
                                          anti_aliasing=True, mode='constant')
            else:
                X[i, :, :, 0] = numpy_from_exr(os.path.join(self.data_folder, file+exr_filext))

            reader = csv.reader(open(os.path.join(self.data_folder, file+'.csv'), "r"), delimiter=" ")
            trajectory_data = list(reader)
            trajectory = np.array(trajectory_data).astype("float")
            if not self.full_trajectory:
                y[i, :] = trajectory[-1]
            else:
                y[i, :, :] = trajectory

        if self.threshold:
            thres = 2
            X = np.where(X >= thres, thres, X)

            if self.feature_scaling:
                X = 2*X/thres-1

        return X, y

    def __len__(self):
        return int(np.floor(len(self.data_file_ids)/self.batch_size))

    def __getitem__(self, item):
        files_to_load = self.data_file_ids[item*self.batch_size:(item+1)*self.batch_size]
        X, y = self._data_generation(files_to_load)
        return X, y

class HangingBinaryDataGenerator(Sequence):
    def __init__(self, data_file_ids, data_folder='.', batch_size=32, dims=(180, 240), resize=True,
                 threshold=True, feature_scaling=True, shuffle=True):
        self.data_file_ids = data_file_ids
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.dims = dims
        self.resize = resize
        self.threshold = threshold
        self.feature_scaling = feature_scaling
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.data_file_ids)

    def _data_generation(self, files_to_load):
        X = np.empty((self.batch_size, self.dims[0], self.dims[1], 1))
        y = np.empty((self.batch_size, 1))

        for i, file in enumerate(files_to_load):
            if self.resize:
                X[i, :, :, 0] = sk_resize(numpy_from_exr(os.path.join(self.data_folder, file+exr_filext)), self.dims,
                                          anti_aliasing=True, mode='constant')
            else:
                X[i, :, :, 0] = numpy_from_exr(os.path.join(self.data_folder, file+exr_filext))

            reader = csv.reader(open(os.path.join(self.data_folder, file+'.csv'), "r"), delimiter=" ")
            trajectory_data = list(reader)
            trajectory = np.array(trajectory_data).astype("float")
            # The next threshold is naive, but better than t < 0.2, as some of the examples are still in midair but
            # will fall down if more time is allowed
            y[i, :] = trajectory[-1] < 0.81

        if self.threshold:
            thres = 2
            X = np.where(X >= thres, thres, X)

            if self.feature_scaling:
                X = 2*X/thres-1

        return X, y

    def __len__(self):
        return int(np.floor(len(self.data_file_ids)/self.batch_size))

    def __getitem__(self, item):
        files_to_load = self.data_file_ids[item*self.batch_size:(item+1)*self.batch_size]
        X, y = self._data_generation(files_to_load)
        return X, y


class HangingImagenetDataGenerator(HangingDataGenerator):
    def __init__(self, data_file_ids, data_folder='.', batch_size=32, resize=True, shuffle=True):
        dims = (224, 224)
        super(HangingImagenetDataGenerator, self).__init__(data_file_ids, data_folder, batch_size, dims,
                                                           resize, shuffle)

    def _data_generation(self, files_to_load):
        X = np.empty((self.batch_size, self.dims[0], self.dims[1], 3))
        y = np.empty((self.batch_size, 3))  # 3 -> x, y, z

        for i, file in enumerate(files_to_load):
            if self.resize:
                X[i, :, :, 0] = sk_resize(numpy_from_exr(os.path.join(self.data_folder, file+exr_filext)), (224, 299),
                                          anti_aliasing=True, mode='constant')[:, 37:262]
            else:
                X[i, :, :, 0] = numpy_from_exr(os.path.join(self.data_folder, file+exr_filext))

            X[i, :, :, 1] = X[i, :, :, 0]
            X[i, :, :, 2] = X[i, :, :, 0]

            reader = csv.reader(open(os.path.join(self.data_folder, file+'.csv'), "r"), delimiter=" ")
            trajectory_data = list(reader)
            trajectory = np.array(trajectory_data).astype("float")
            y[i, :] = trajectory[-1]

        if self.threshold:
            thres = 2
            X = np.where(X >= thres, thres, X)

            if self.feature_scaling:
                X = 2*X/thres-1

        return X, y