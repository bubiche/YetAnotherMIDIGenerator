import numpy as np
import pretty_midi
import common_config


def output_midi_file_from_note_list(note_list,
                                    output_file_path, note_numerizer,
                                    sampling_frequency=common_config.SAMPLING_FREQUENCY_OUTPUT,
                                    start_idx=common_config.SLIDING_WINDOW_SIZE - 2, # start from the last note of the seed
                                    n_note_generate=common_config.N_NOTE_GENERATE):
    note_string_list = [note_numerizer.note_string_by_number[num] for num in note_list]
    # 0 at the start + last note of the seed + n_note_generate generated notes
    piano_roll = np.zeros((128, n_note_generate + 2), dtype=np.int8)

    # create the piano roll
    print('Create the piano roll')
    for time_idx, note_string in enumerate(note_string_list[start_idx:]):
        if note_string == common_config.SILENT_CHAR:
            continue

        splitted_note = note_string.split(',')
        for i in splitted_note:
            piano_roll[int(i)][time_idx] = 1

    # create pretty_midi object from piano roll
    n_notes, n_time_frames = piano_roll.shape
    pretty_midi_obj = pretty_midi.PrettyMIDI()
    # channel 0 for piano
    instrument = pretty_midi.Instrument(program=0)

    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track of velocity and note start time
    velocity_by_note = np.zeros(n_notes, dtype=int)
    start_time_by_note = np.zeros(n_notes)

    print('Finalizing MIDI data')
    for time, note in zip(*velocity_changes):
        # use time + 1 because we did some padding above
        velocity = piano_roll[note, time + 1]
        time = time / sampling_frequency
        if velocity > 0:
            if velocity_by_note[note] == 0:
                start_time_by_note[note] = time
                velocity_by_note[note] = velocity
        else:
            prerry_midi_note = pretty_midi.Note(
                velocity=velocity_by_note[note],
                pitch=note,
                start=start_time_by_note[note],
                end=time
            )
            instrument.notes.append(prerry_midi_note)
            velocity_by_note[note] = 0
    pretty_midi_obj.instruments.append(instrument)

    estimated_tempo = int(pretty_midi_obj.estimate_tempo())
    print('Estimated Tempo {}'.format(estimated_tempo))
    for note in pretty_midi_obj.instruments[0].notes:
        note.velocity = estimated_tempo

    # write to file
    pretty_midi_obj.write(output_file_path)
