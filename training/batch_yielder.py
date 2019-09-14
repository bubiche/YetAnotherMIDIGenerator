import numpy as np
import h5py
import random

import common_config


class BatchYielder(object):
    def __init__(self,
                 input_file_name=common_config.INPUT_FILE_NAME, input_dataset_key=common_config.INPUT_HDF5_KEY,
                 target_file_name=common_config.TARGET_FILE_NAME, target_dataset_key=common_config.TARGET_HDF5_KEY,
                 batch_size=common_config.BATCH_SIZE, epoch_count=common_config.EPOCH_COUNT):
        self._batch_size = batch_size
        self._epoch_count = epoch_count
        
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
        for i in range(self._epoch_count):
            print('Epoch Number {}'.format(i))
            self.shuffle_data()
            for b in range(self.batch_per_epoch):
                # yield these
                x_batch = list()
                y_batch = list()

                for j in range(b * self.batch_size, b * self.batch_size + self.batch_size):
                    if j >= self.n_train:
                        continue
                    x_instance = self.get_input_at_index(self.shuffle_idx[j])
                    y_instance = self.get_target_at_index(self.shuffle_idx[j])

                    x_batch.append(x_instance)
                    y_batch.append(y_instance)

                yield x_batch, y_batch

    def next_epoch(self):
        x_batch = list()
        y_batch = list()

        for j in range(self.n_train):
            x_instance = self.get_input_at_index(self.shuffle_idx[j])
            y_instance = self.get_target_at_index(self.shuffle_idx[j])

            x_batch.append(x_instance)
            y_batch.append(y_instance)

        return np.array(x_batch), np.array(y_batch)
