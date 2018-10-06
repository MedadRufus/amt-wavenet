'''Utility script that trims MIDI and WAV files to specified time segment.'''

from __future__ import print_function

import sys
import librosa
import pretty_midi
import os, fnmatch
import argparse
import logging as log

sys.path.append(os.path.join(os.getcwd(), os.pardir))
from utils import find_files, roll_encode, get_roll_index

LOG_LEVEL = log.DEBUG


def get_arguments():
    '''Parses script arguments.'''

    parser = argparse.ArgumentParser(
        description='Trims MIDI and WAV files to specified time range')
    parser.add_argument(
        '--input_dir', type=str, help='The directory with input MIDI files.')
    parser.add_argument(
        '--output_dir', type=str,
        help='The directory where trimmed MIDI files should be stored.')
    parser.add_argument(
        '--sec_from', type=float, help='Start time [s] to trim the files from.')
    parser.add_argument(
        '--sec_to', type=float, help='End time [s] to trim the files to.')
    return parser.parse_args()


def trim_midi_file(s_from, s_to, midi):
    '''Performs trimming on PrettyMIDI object.'''

    new_midi = pretty_midi.PrettyMIDI()
    for instrument in midi.instruments:
        new_inst = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name)
        for note in instrument.notes:
            if (note.start >= s_from and
                note.start < s_to):
                # shift start time back
                note.start = note.start - s_from
                note.end = note.end - s_from
                # cut note durating through limit
                if note.end > s_to - s_from:
                    note.end = s_to - s_from
                # add to collection
                new_note = pretty_midi.Note(
                    start=note.start,
                    end=note.end,
                    pitch=note.pitch,
                    velocity=note.velocity)
                new_inst.notes.append(new_note)
        new_midi.instruments.append(new_inst)
    return new_midi

def trim_audio_file(s_from, s_to, audio, sr):
    '''Performs trimming on audio snippet'''

    return audio[int(sr*s_from):int(sr*s_to)]


def main():
    args = get_arguments()

    log.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                    level=LOG_LEVEL)
    log.info("Start of '{}'.".format(__file__))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        log.info("Created output directory '{}'.".format(args.output_dir))

    for f in find_files(args.input_dir, '*.mid', path=False):
        midi_old = pretty_midi.PrettyMIDI(
            os.path.join(args.input_dir, f))
        midi_new = trim_midi_file(args.sec_from, args.sec_to, midi_old)
        midi_new.write(
            os.path.join(args.output_dir, f))
        log.info("Processed file '{}'".format(f))

    for f in find_files(args.input_dir, '*.wav', path=False):
        audio_old, sr = librosa.load(os.path.join(args.input_dir, f))
        audio_new = trim_audio_file(args.sec_from, args.sec_to, audio_old, sr)
        librosa.output.write_wav(
            os.path.join(args.output_dir, f), audio_new, sr)
        log.info("Processed file '{}'".format(f))

    log.info("End of '{}'.".format(__file__))


if __name__ == '__main__':
    main()
