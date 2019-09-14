import argparse
import os

import midi_processor
import common_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--preprocess_folder', help='preprocess MIDI files in specified folder')
    group.add_argument('--clean', help='clean preprocessed data', action='store_true')

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
