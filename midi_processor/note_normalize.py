import common_config


def normalize_note(note):
    if note != common_config.SILENT_NOTE:
        note = max(note, common_config.MIN_NOTE)
        note = min(note, common_config.MAX_NOTE)
    return int(note)