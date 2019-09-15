import pickle
import common_config


class NoteNumerizer(object):
    """transform a string representing a bunch of notes played together, e.g. "20,43" into a number"""
    number_by_note_string = {}
    note_string_by_number = {}
    note_string_count = 0

    def add_note_string(self, note_string):
        if note_string in self.number_by_note_string:
            return

        self.note_string_count += 1
        self.number_by_note_string[note_string] = self.note_string_count
        self.note_string_by_number[self.note_string_count] = note_string

    def save_to_pickle(self, save_file_name=common_config.NOTE_AND_NUMBER_MAPPER_FILE_NAME):
        with open(save_file_name, 'wb') as save_file:
            pickle.dump(self.number_by_note_string, save_file)

    def load_from_pickle(self, load_file_name=common_config.NOTE_AND_NUMBER_MAPPER_FILE_NAME):
        with open(load_file_name, 'rb') as load_file:
            self.number_by_note_string = pickle.load(load_file)
            for note_string, number in self.number_by_note_string.items():
                self.note_string_by_number[number] = note_string

            self.note_string_count = len(self.note_string_by_number)
