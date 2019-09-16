import numpy as np
import h5py
import random

import common_config


class BatchYielder(object):
    def __init__(self,
                 input_file_name=common_config.INPUT_FILE_NAME, input_dataset_key=common_config.INPUT_HDF5_KEY,
                 target_file_name=common_config.TARGET_FILE_NAME, target_dataset_key=common_config.TARGET_HDF5_KEY,
                 batch_size=common_config.BATCH_SIZE):
        self._batch_size = batch_size

        self.input_file = h5py.File(input_file_name, 'r')
        self.target_file = h5py.File(target_file_name, 'r')

        self.input_dataset = self.input_file[input_dataset_key]
        self.target_dataset = self.target_file[target_dataset_key]

        self.n_train = self.input_dataset.shape[0]
        
        if self._batch_size > self.n_train:
            self._batch_size = self.n_train
        self.batch_per_epoch = int(np.ceil(self.n_train / self._batch_size))

    def shuffle_data(self):
        self.shuffle_idx = np.random.permutation(self.n_train)
            
    def get_input_at_index(self, idx):
        return self.input_dataset[idx]

    def get_target_at_index(self, idx):
        return self.target_dataset[idx]

    def next_batch(self):
        b = 0
        self.shuffle_data()
        while True:
            # yield these
            x_batch = []
            y_batch = []

            for j in range(b * self._batch_size, b * self._batch_size + self._batch_size):
                if j >= self.n_train:
                    continue
                x_instance = self.get_input_at_index(self.shuffle_idx[j])
                y_instance = self.get_target_at_index(self.shuffle_idx[j])

                x_batch.append(x_instance)
                y_batch.append(y_instance)

            b += 1
            # end of current epoch
            if b >= self.batch_per_epoch:
                b = 0
                self.shuffle_data()
            yield np.array(x_batch), np.array(y_batch)
