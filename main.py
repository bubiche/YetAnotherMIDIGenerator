import argparse
import os
import pickle
import numpy as np

import midi_processor
import common_config
import nnet


def get_unique_notes_count(load_file_name=common_config.NOTE_AND_NUMBER_MAPPER_FILE_NAME):
    number_by_note_string = {}
    with open(load_file_name, 'rb') as load_file:
        number_by_note_string = pickle.load(load_file)
    return len(number_by_note_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args_group = parser.add_mutually_exclusive_group()
    args_group.add_argument('-p', '--preprocess_folder', help='preprocess MIDI files in specified folder')
    args_group.add_argument('--clean', help='clean preprocessed data', action='store_true')
    args_group.add_argument('-t', '--train', help='train the model', action='store_true')
    args_group.add_argument('--train_from_ckpt', help='train the model starting from the checkpoint specified in path')
    args_group.add_argument('--generate_random', help='generate random music from WEIGHTS_FILE and output to OUTPUT_FILE', nargs=2, metavar=('WEIGHTS_FILE', 'OUTPUT_FILE'))
    args_group.add_argument('--generate_from_seed', help='generate music using SEED_NOTE from WEIGHTS_FILE and output to OUTPUT_FILE', nargs=3, metavar=('SEED_NOTE', 'WEIGHTS_FILE', 'OUTPUT_FILE'))

    args = parser.parse_args()
    if args.preprocess_folder:
        midi_processor.preprocess_training_data(folder_path=args.preprocess_folder)
    elif args.clean:
        FILES_TO_REMOVE_LIST = [
            common_config.INPUT_FILE_NAME, common_config.TARGET_FILE_NAME,
            common_config.NOTE_AND_NUMBER_MAPPER_FILE_NAME
        ]

        for file_to_remove in FILES_TO_REMOVE_LIST:
            if os.path.exists(file_to_remove):
                os.remove(file_to_remove)
    elif args.train or args.train_from_ckpt:
        unique_notes_count = get_unique_notes_count()
        print('Training with unique notes: {}'.format(unique_notes_count))
        print('Build the net')
        midi_net = nnet.MIDINet(unique_notes_count=unique_notes_count)
        print('Model summary')
        midi_net.print_model_summary()
        if args.train_from_ckpt:
            print('Loading weights')
            midi_net.load_weights(args.train_from_ckpt)
        print('Start training')
        midi_net.train()
    elif args.generate_random or args.generate_from_seed:
        note_numerizer = midi_processor.NoteNumerizer()
        note_numerizer.load_from_pickle()
        unique_notes_count = note_numerizer.note_string_count
        midi_net = nnet.MIDINet(unique_notes_count=unique_notes_count)
        if args.generate_random:
            print('Generate random music')
            first_input = np.random.randint(0, unique_notes_count, common_config.SLIDING_WINDOW_SIZE).tolist()
            weights_file_path = args.generate_random[0]
            output_file_path = args.generate_random[1]
        elif args.generate_from_seed:
            print('Generate music from seed # {}'.format(args.generate_from_seed[0]))
            first_input = [note_numerizer.number_by_note_string[common_config.SILENT_CHAR] for i in range(common_config.SLIDING_WINDOW_SIZE)]
            first_input.append(note_numerizer.number_by_note_string[args.generate_from_seed[0]])
            weights_file_path = args.generate_from_seed[1]
            output_file_path = args.generate_random[2]

        print('Loading weights')
        midi_net.load_weights(weights_file_path)

        print('Generating notes')
        note_list = first_input
        for i in range(common_config.N_NOTE_GENERATE):
            # input of the network is (None, window_size)
            cur_input = np.array([note_list])[:, i:i + common_config.SLIDING_WINDOW_SIZE]
            prediction = midi_net.predict(cur_input)
            note_list.append(prediction)

        print('Save to file')
        midi_processor.output_midi_file_from_note_list(note_list, output_file_path, note_numerizer)
