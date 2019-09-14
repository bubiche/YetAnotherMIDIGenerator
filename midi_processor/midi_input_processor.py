import pretty_midi
import numpy as np
import h5py
import os

import common_config
from .note_numerize import NoteNumerizer


# generate a pair of input - target list from MIDI file
def midiToInputTargetPair(file_path, note_numerizer,
                          sampling_frequency=common_config.SAMPLING_FREQUENCY_PREPROCESS,
                          sliding_window_size=common_config.SLIDING_WINDOW_SIZE,
                          silent_char=common_config.SILENT_CHAR):
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

    # notes_by_time[t] = array of notes played at time t
    notes_by_time = {}
    start_time = times_list[0]
    end_time = times_list[0]
    for time_idx in times_list:
        notes = indices[0][np.where(indices[1] == time_idx)]
        note_string = ','.join(str(note) for note in notes)
        note_numerizer.add_note_string(note_string)
        notes_by_time[time_idx] = note_numerizer.number_by_note_string[note_string]
        if time_idx < start_time:
            start_time = time_idx
        if time_idx > end_time:
            end_time = time_idx

    # Create the input - target pairs
    # we look at sliding_window_size each time
    # input is current window, target is the next note
    # fill silent (empty) times with the character silent_char
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
                cur_input.append(note_numerizer.number_by_note_string[silent_char])
                is_append_target = True

        for i in range(start_idx, sliding_window_size):
            cur_time = time_idx - (sliding_window_size - i - 1)
            if cur_time in notes_by_time:
                cur_input.append(notes_by_time[cur_time])
            else:
                cur_input.append(note_numerizer.number_by_note_string[silent_char])

        # create target from next window at time_idx + 1
        if (time_idx + 1) in notes_by_time:
            cur_target = notes_by_time[time_idx + 1]
        else:
            cur_target = note_numerizer.number_by_note_string[silent_char]
        input_list.append(cur_input)
        target_list.append(cur_target)

    return input_list, target_list


# read all MIDI files in a folder, preprocess them into inputs and targets
# save the result to hdf5 files
def preprocessTrainingData(folder_path='midi_files',
                          input_save_file_name=common_config.INPUT_FILE_NAME, input_dataset_key=common_config.INPUT_HDF5_KEY,
                          target_save_file_name=common_config.TARGET_FILE_NAME, target_dataset_key=common_config.TARGET_HDF5_KEY,
                          numerizer_file_name=common_config.NOTE_AND_NUMBER_MAPPER_FILE_NAME,
                          silent_char=common_config.SILENT_CHAR,
                          sliding_window_size=common_config.SLIDING_WINDOW_SIZE):
    note_numerizer = NoteNumerizer()
    note_numerizer.add_note_string(silent_char)

    input_list = []
    target_list = []
    file_count = 0
    print('Reading MIDI files')
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                file_count += 1
                if file_count % 10 == 0:
                    print('Processed {} file'.format(file_count))
                file_path = os.path.join(root, file)
                cur_input_list, cur_target_list = midiToInputTargetPair(file_path=file_path, note_numerizer=note_numerizer)
                input_list.extend(cur_input_list)
                target_list.extend(cur_target_list)

    print('Saving note - number map')
    note_numerizer.save_to_pickle(save_file_name=numerizer_file_name)

    print('Saving hdf5 files')
    data_size = len(input_list)
    input_file = h5py.File(input_save_file_name, 'w')
    target_file = h5py.File(target_save_file_name, 'w')

    input_dataset = input_file.create_dataset(input_dataset_key, (data_size, sliding_window_size), dtype='f')
    target_dataset = target_file.create_dataset(target_dataset_key, (data_size,), dtype='f')

    for i in range(data_size):
        if i % 20 == 0:
            print('Saved {} pairs'.format(i + 1))
        input_dataset[i] = input_list[i]
        target_dataset[i] = target_list[i]

    input_file.close()
    target_file.close()