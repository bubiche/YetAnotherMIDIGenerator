import pretty_midi
import numpy as np
import h5py
import os

import common_config
from .note_normalize import normalize_note

# generate a pair of input - target list from MIDI file
def midi_to_input_target_pair(file_path,
                              sampling_frequency=common_config.SAMPLING_FREQUENCY_PREPROCESS,
                              sliding_window_size=common_config.SLIDING_WINDOW_SIZE,
                              silent_note=common_config.SILENT_NOTE):
    pretty_midi_file = pretty_midi.PrettyMIDI(file_path)
    # piano channel
    piano_midi = pretty_midi_file.instruments[0]
    # piano_roll's dim is notes x time
    # (i.e. 128 notes = 128 rows,
    # each row has length equal to time depending on our sampling frequency)
    piano_roll = piano_midi.get_piano_roll(fs=sampling_frequency)

    # times when there is a note played (aka has value > 0)
    times_list = np.unique(np.where(piano_roll > 0)[1])
    indices = np.where(piano_roll > 0)

    # note_by_time[t] = highest note played at time t
    note_by_time = {}
    start_time = times_list[0]
    end_time = times_list[0]
    for time_idx in times_list:
        notes = indices[0][np.where(indices[1] == time_idx)]
        note_by_time[time_idx] = normalize_note(max(notes))
        if time_idx < start_time:
            start_time = time_idx
        if time_idx > end_time:
            end_time = time_idx

    # Create the input - target pairs
    # we look at sliding_window_size each time
    # input is current window, target is the next note
    # fill silent (empty) times with the character silent_note
    input_list = []
    target_list = []
    for idx, time_idx in enumerate(range(start_time, end_time)):
        cur_input = []
        cur_target = 0

        # create input from current window at time_idx
        start_idx = 0
        is_append_target = False
        if idx < sliding_window_size:
            start_idx = sliding_window_size - idx - 1
            for _ in range(start_idx):
                cur_input.append(silent_note)
                is_append_target = True

        for i in range(start_idx, sliding_window_size):
            cur_time = time_idx - (sliding_window_size - i - 1)
            cur_input.append(note_by_time.get(cur_time, silent_note))

        # create target from next window at time_idx + 1
        # target is a 1x1 tensor
        cur_target = [note_by_time.get(time_idx + 1, silent_note)]

        input_list.append(cur_input)
        target_list.append(cur_target)

    return input_list, target_list


# read all MIDI files in a folder, preprocess them into inputs and targets
# save the result to hdf5 files
def preprocess_training_data(folder_path='midi_files',
                             input_save_file_name=common_config.INPUT_FILE_NAME, input_dataset_key=common_config.INPUT_HDF5_KEY,
                             target_save_file_name=common_config.TARGET_FILE_NAME, target_dataset_key=common_config.TARGET_HDF5_KEY,
                             silent_note=common_config.SILENT_NOTE,
                             sliding_window_size=common_config.SLIDING_WINDOW_SIZE):

    max_dataset_size = 65536
    input_file = h5py.File(input_save_file_name, 'w')
    target_file = h5py.File(target_save_file_name, 'w')
    input_dataset = input_file.create_dataset(input_dataset_key, (max_dataset_size, sliding_window_size), maxshape=(None, sliding_window_size), dtype='f')
    target_dataset = target_file.create_dataset(target_dataset_key, (max_dataset_size, 1), maxshape=(None, 1), dtype='f')

    file_count = 0
    input_target_pair_count = 0
    print('Reading MIDI files')
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                file_count += 1
                if file_count % 10 == 0:
                    print('Processed {} file'.format(file_count))
                file_path = os.path.join(root, file)
                cur_input_list, cur_target_list = midi_to_input_target_pair(file_path=file_path)
                for input_value, target_value in zip(cur_input_list, cur_target_list):
                    input_dataset[input_target_pair_count] = input_value
                    target_dataset[input_target_pair_count] = target_value
                    input_target_pair_count += 1

                    # increase capacity of datasets when almost full
                    if input_target_pair_count >= max_dataset_size * 0.9:
                        max_dataset_size *= 2
                        input_dataset.resize((max_dataset_size, sliding_window_size))
                        target_dataset.resize((max_dataset_size, 1))

    input_dataset.resize((input_target_pair_count, sliding_window_size))
    target_dataset.resize((input_target_pair_count, 1))

    input_file.close()
    target_file.close()
