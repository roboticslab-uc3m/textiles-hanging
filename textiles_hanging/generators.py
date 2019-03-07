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


class HangingDataGenerator(Sequence):
    def __init__(self, data_file_ids, data_folder='.', batch_size=32, dims=(180, 240), resize=True,
                 shuffle=True):
        self.data_file_ids = data_file_ids
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.dims = dims
        self.resize = resize
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.data_file_ids)

    def _data_generation(self, files_to_load):
        X = np.empty((self.batch_size, *self.dims, 1))
        y = np.empty((self.batch_size, 3))  # 3 -> x, y, z

        for i, file in enumerate(files_to_load):
            if self.resize:
                X[i, :, :, 0] = sk_resize(numpy_from_exr(os.path.join(self.data_folder, file+'.exr0040.exr')), self.dims,
                                          anti_aliasing=True)
            else:
                X[i, :, :, 0] = numpy_from_exr(os.path.join(self.data_folder, file+'.exr0040.exr'))

            reader = csv.reader(open(os.path.join(self.data_folder, file+'.csv'), "r"), delimiter=" ")
            trajectory_data = list(reader)
            trajectory = np.array(trajectory_data).astype("float")
            y[i, :] = trajectory[-1]

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
        X = np.empty((self.batch_size, *self.dims, 3))
        y = np.empty((self.batch_size, 3))  # 3 -> x, y, z

        for i, file in enumerate(files_to_load):
            if self.resize:
                X[i, :, :, 0] = sk_resize(numpy_from_exr(os.path.join(self.data_folder, file+'exr0040.exr')), (224, 299),
                                          anti_aliasing=True)[:, 37:262]
                X[i, :, :, 1] = sk_resize(numpy_from_exr(os.path.join(self.data_folder, file+'exr0040.exr')), (224, 299),
                                          anti_aliasing=True)[:, 37:262]
                X[i, :, :, 2] = sk_resize(numpy_from_exr(os.path.join(self.data_folder, file+'exr0040.exr')), (224, 299),
                                          anti_aliasing=True)[:, 37:262]
            else:
                X[i, :, :, 0] = numpy_from_exr(os.path.join(self.data_folder, file+'exr0040.exr'))
                X[i, :, :, 1] = numpy_from_exr(os.path.join(self.data_folder, file+'exr0040.exr'))
                X[i, :, :, 2] = numpy_from_exr(os.path.join(self.data_folder, file+'exr0040.exr'))

            reader = csv.reader(open(os.path.join(self.data_folder, file+'.csv'), "r"), delimiter=" ")
            trajectory_data = list(reader)
            trajectory = np.array(trajectory_data).astype("float")
            y[i, :] = trajectory[-1]

        return X, y