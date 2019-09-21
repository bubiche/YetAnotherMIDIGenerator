import os
import datetime
import tensorflow as tf
import numpy as np

import common_config
from .batch_yielder import BatchYielder
from .custom_functions import swish


# https://github.com/keras-team/keras/issues/2850
class NBatchLogger(tf.keras.callbacks.Callback):
    def __init__(self, total_batch, n_batch=10, batch_size=common_config.BATCH_SIZE):
        # n_batch is how many batch to wait before output something
        self.seen = 0
        self._n_batch = n_batch
        self._total_batch = total_batch
        self._batch_size = batch_size

    def on_batch_end(self, batch, logs={}):
        self.seen += 1
        if  self.seen >= self._total_batch:
            self.seen = 1
        if  self.seen % self._n_batch == 0:
            print('\n{}/{} Batches - loss: {}'.format(self.seen, self._total_batch, logs.get('loss')))


class MIDINet(object):
    def __init__(self, unique_notes_count, 
                 sliding_window_size=common_config.SLIDING_WINDOW_SIZE, name=common_config.MODEL_NAME,
                 input_file_name=common_config.INPUT_FILE_NAME, input_dataset_key=common_config.INPUT_HDF5_KEY,
                 target_file_name=common_config.TARGET_FILE_NAME, target_dataset_key=common_config.TARGET_HDF5_KEY,
                 batch_size=common_config.BATCH_SIZE, epoch_count=common_config.EPOCH_COUNT):
        self._unique_notes_count = unique_notes_count
        self._sliding_window_size = sliding_window_size
        self._name = name
        self._batch_size = batch_size
        self._batch_yielder = BatchYielder(
            input_file_name=input_file_name, input_dataset_key=input_dataset_key,
            target_file_name=target_file_name, target_dataset_key=target_dataset_key,
            batch_size=batch_size
        )

        self._epoch_count = epoch_count
        self._model = self.build_model()

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(self._sliding_window_size,))

        reshape = tf.keras.layers.Reshape((self._sliding_window_size, 1))(inputs)
        conv1d = tf.keras.layers.Conv1D(64, kernel_size=4, activation='hard_sigmoid')(reshape)
        flatten = tf.keras.layers.Flatten()(conv1d)

        dense1 = tf.keras.layers.Dense(128)(flatten)
        leaky_relu = tf.keras.layers.LeakyReLU()(dense1)

        dense2 = tf.keras.layers.Dense(256, activation=swish)(leaky_relu)
        drop_out = tf.keras.layers.Dropout(0.2)(dense2)
        # soft max help make values go closer to upper/lower bound
        outputs = tf.keras.layers.Dense(self._unique_notes_count + 1, activation='softmax')(drop_out)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy')
        return model

    def save_weights(self, save_folder=common_config.FINAL_WEIGHT_FOLDER):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self._model.save_weights('./{}/{}-nc{}-sw{}.h5'.format(save_folder, timestamp, self._unique_notes_count, self._sliding_window_size))

    def load_weights(self, load_path):
        self._model.load_weights(load_path)

    def train(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        checkpoint_path = '{}/cp-{}-nc{}-sw{}-'.format(common_config.CHECKPOINT_FOLDER, timestamp, self._unique_notes_count, self._sliding_window_size) + '-{epoch:04d}.hdf5'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_weights_only=True
        )
        steps_per_epoch = self._batch_yielder.n_train // self._batch_size
        n_batch_callback = NBatchLogger(total_batch=steps_per_epoch, n_batch=10, batch_size=self._batch_size)
        self._model.fit_generator(self._batch_yielder.next_batch(), epochs=self._epoch_count, callbacks=[checkpoint_callback, n_batch_callback], steps_per_epoch=steps_per_epoch, verbose=2)
        print('Saving final weights')
        self.save_weights()

    def print_model_summary(self):
        print(self._model.summary())

    # predict the next note
    def predict(self, intput_notes):
        # input_notes must be a numpy array with the shape of the net's input
        return np.random.choice(self._unique_notes_count + 1, replace=False, p=self._model.predict(intput_notes)[0])
