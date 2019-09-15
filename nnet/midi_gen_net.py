import os
import datetime
import tensorflow as tf

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
        self.seen += logs.get('size', 0)
        if (self.seen / self._batch_size) % self._n_batch == 0:
            print('\n{}/{} Batches'.format(self.seen, self._total_batch))


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
        # self._unique_notes_count + 1 because the first note is 1 not 0, 128 is chosen because there were 128 notes in pretty_midi output
        embedding = tf.keras.layers.Embedding(self._unique_notes_count + 1, output_dim=64, input_length=self._sliding_window_size)(inputs)

        dense1 = tf.keras.layers.Dense(128)(embedding)
        leaky_relu = tf.keras.layers.LeakyReLU()(dense1)
        gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, recurrent_dropout=0.2))(leaky_relu)

        drop_out = tf.keras.layers.Dropout(0.4)(gru)
        dense2 = tf.keras.layers.Dense(512, activation=swish)(drop_out)
        # soft max help make values go closer to upper/lower bound
        outputs = tf.keras.layers.Dense(self._unique_notes_count + 1, activation='softmax')(dense2)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='sparse_categorical_crossentropy')
        return model

    def save_weights(self, save_folder=common_config.FINAL_WEIGHT_FOLDER):
        timestamp = datetime.datetime.now().strftime('%Y/%m/%d_%H:%M:%S')
        self._model.save_weights('./{}/{}-nc{}-sw{}.weights'.format(save_folder, timestamp, self._unique_notes_count, self._sliding_window_size))

    def load_weights(self, load_path):
        self._model.load_weights(load_path)

    def train(self):
        timestamp = datetime.datetime.now().strftime('%Y/%m/%d_%H:%M:%S')
        checkpoint_path = '{}/cp-{}-nc{}-sw{}-'.format(common_config.CHECKPOINT_FOLDER, timestamp, self._unique_notes_count, self._sliding_window_size) + '-{epoch:04d}.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, verbose=1, save_weights_only=True
        )
        steps_per_epoch = self._batch_yielder.n_train // self._batch_size
        n_batch_callback = NBatchLogger(total_batch=steps_per_epoch, n_batch=50, batch_size=self._batch_size)
        self._model.fit_generator(self._batch_yielder.next_batch(), epochs=self._epoch_count, callbacks=[checkpoint_callback, n_batch_callback], steps_per_epoch=steps_per_epoch, verbose=2)
        print('Saving final weights')
        self.save_weights()

    def print_model_summary(self):
        print(self._model.summary())
