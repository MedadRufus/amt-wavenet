'''Utility script that converts CSV note labels to MIDI files.'''

from __future__ import print_function

import sys
import csv
import numpy as np
import pretty_midi
import os, fnmatch
import argparse
import logging as log

sys.path.append(os.path.join(os.getcwd(), os.pardir))
from utils import find_files, roll_encode, get_roll_index

LOG_LEVEL = log.DEBUG


def get_arguments():
    '''Parses script arguments.'''

    parser = argparse.ArgumentParser(description='Converts CSV to MIDI labels.')
    parser.add_argument(
        '--input_dir', type=str,
        help='The directory with input CSV files.')
    parser.add_argument(
        '--output_dir', type=str,
        help='The directory where output MIDI files should be stored.')
    parser.add_argument(
        '--audio_sr', type=int, default=44100,
        help='Sampling rate for time events conversion'
        'from [samples] to [seconds].')
    return parser.parse_args()


def csv2mid(csv_data, sr):
    '''Creates PrettyMIDI object from CSV data.'''

    new_midi = pretty_midi.PrettyMIDI()

    instrums = {}
    for i in np.unique([int(l[4]) for l in csv_data]):
        instrums[i] = pretty_midi.Instrument(program=i)

    for l in csv_data:
        start = float(int(l[0])/sr)
        end = float(int(l[2])/sr)
        pitch = int(l[6])
        note = pretty_midi.Note(start=start, end=end, pitch=pitch, velocity=127)
        instrums[int(l[4])].notes.append(note)

    midi = pretty_midi.PrettyMIDI()
    for instrument in instrums.values():
        midi.instruments.append(instrument)

    return midi


def main():
    args = get_arguments()

    log.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                    level=LOG_LEVEL)
    log.info("Start of '{}'.".format(__file__))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        log.info("Created output directory '{}'.".format(args.output_dir))

    for fname in find_files(args.input_dir, '*.csv', path=False):
        with open(os.path.join(args.input_dir, fname), 'r') as f:
            csv_data = list(csv.reader(f, delimiter='\t'))
        midi = csv2mid(csv_data[1:], args.audio_sr)
        midi.write(os.path.join(args.output_dir, fname.replace('.csv', '.mid')))
        log.info("Processed file '{}'".format(fname))

    log.info("End of '{}'.".format(__file__))


if __name__ == '__main__':
    main()
