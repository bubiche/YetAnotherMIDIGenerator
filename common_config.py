SILENT_CHAR = 's'

# sampling frequency for pretty_midi
# from their document: http://craffel.github.io/pretty-midi/#pretty_midi.PrettyMIDI.get_piano_roll
# "Sampling frequency of the columns, i.e. each column is spaced apart by 1./fs seconds"
SAMPLING_FREQUENCY_PREPROCESS = 30

# size of each input instance (x)
SLIDING_WINDOW_SIZE = 40

INPUT_FILE_NAME = 'input.hdf5'
INPUT_HDF5_KEY = 'input'
TARGET_FILE_NAME = 'target.hdf5'
TARGET_HDF5_KEY = 'target'

NOTE_AND_NUMBER_MAPPER_FILE_NAME = 'number_by_note_string.pickle'

# batch size == how many x - y pair will be processed in each batch
# preferably to be a power of 2
BATCH_SIZE = 256
EPOCH_COUNT = 4

MODEL_NAME = 'midi_gen_net'

CHECKPOINT_FOLDER = 'ckpt'
FINAL_WEIGHT_FOLDER = 'weights'