import argparse
import os
import pickle
import numpy as np

import midi_processor
import common_config
import nnet


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
            common_config.INPUT_FILE_NAME, common_config.TARGET_FILE_NAME
        ]

        for file_to_remove in FILES_TO_REMOVE_LIST:
            if os.path.exists(file_to_remove):
                os.remove(file_to_remove)
    elif args.train or args.train_from_ckpt:
        print('Build the net')
        midi_net = nnet.MIDINet()
        print('Model summary')
        midi_net.print_model_summary()
        if args.train_from_ckpt:
            print('Loading weights')
            midi_net.load_weights(args.train_from_ckpt)
        print('Start training')
        midi_net.train()
    elif args.generate_random or args.generate_from_seed:
        midi_net = nnet.MIDINet()
        if args.generate_random:
            print('Generate random music')
            first_input_rand = np.random.randint(0, common_config.MAX_NOTE + 1, common_config.SLIDING_WINDOW_SIZE).tolist()
            first_input = [midi_processor.normalize_note(i) for i in first_input_rand]
            weights_file_path = args.generate_random[0]
            output_file_path = args.generate_random[1]
        elif args.generate_from_seed:
            print('Generate music from seed # {}'.format(args.generate_from_seed[0]))
            first_input = [common_config.SILENT_NOTE for i in range(common_config.SLIDING_WINDOW_SIZE - 1)]
            first_input.append(args.generate_from_seed[0])
            weights_file_path = args.generate_from_seed[1]
            output_file_path = args.generate_from_seed[2]

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
        midi_processor.output_midi_file_from_note_list(note_list, output_file_path)
