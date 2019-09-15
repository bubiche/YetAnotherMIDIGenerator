import argparse
import os
import pickle

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
