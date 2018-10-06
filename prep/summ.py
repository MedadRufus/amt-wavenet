'''Generates summary of dataset parameters and properties. Adjusted for
datasets generated from MIDI data using tansposition and synthesis scripts.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import argparse
import pretty_midi
from collections import OrderedDict
import json
import re

sys.path.append(os.path.join(os.getcwd(), os.pardir))
from utils import find_files, roll_encode, get_roll_index

DATA_DIRECTORY = '../data/sanitycheck'
MIN_FLOAT = sys.float_info.min


def get_arguments():
    '''Parses script arguments.'''

    parser = argparse.ArgumentParser(
        description='Generates summary of dataset parameters and properties')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the dataset files.')
    return parser.parse_args()


def units_format(seconds):
    '''Encodes time from seconds to daytime format and other
    units. Returns values as OrderedDict().'''

    units = OrderedDict()
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    units['daytime'] = "| {:} hours | {:02d} minutes | {:02d} seconds |".format(
        int(h), int(m), int(s))
    units['seconds'] = seconds
    units['minutes'] = seconds / 60
    units['hours'] = seconds / 60**2

    return units


def update_poly_stats(polyphony, midi):
    '''Parses midi file and updates polyphony stats.'''

    proll = midi.get_piano_roll(fs=100)
    proll[proll > 0] = 1

    table, indices = roll_encode(proll, 1)

    unique, counts = np.unique(indices, return_counts=True)
    occurred = dict(zip(unique, counts))

    for roll in table:
        degree = int(sum(roll))
        index = get_roll_index(table, roll)[0]
        polyphony[degree] += occurred[index]

    return polyphony


def get_stats(data_dir):
    '''Calculates statistics from directory contents.

    Args:
        data_dir: Path to directory containing dataset
        to be analyzed.

    Returns:
        A python dictionary `stats` with keys and values described as follows.

        *   `n_mids`: Number of `mid` files found in `data_dir`.
        *   `n_wavs`: Number of `wav` files found in `data_dir`.
        *   `n_fonts`: Number of soundfonts used for generation, assumed from
            and calculated as proportion of `wav` to `mid` files.
        *   `time_tunes`: Duration [s] of tunes based on midi events, with
            subfields for `total` dataset duration, `avg` tune duration,
            `max` tune duration and `min` tune duration.
        *   `notes`: Average note duration [s] `time_avg` and total count of
            notes in data set `n_total`.
        *   `time_total_notes`: Total duration [s] of individual notes. Vector
            of 128 floats.
        *   `n_total_notes`: Total number of occurrences of individual notes.
            Vector of 128 floats.
        *   `time_avg_notes`: Average durations [s] of individual notes.
        *   `n_avg_notes`: Average counts of individual notes in singe tune.
    '''

    mids = find_files(data_dir, '*.mid')
    wavs = find_files(data_dir, '*.wav')
    if not mids or not wavs:
        raise ValueError("No mid/wav files found "
                         "in '{}'.".format(data_dir))

    stats = OrderedDict()
    stats['n_mids'] = len(mids)
    stats['n_wavs'] = len(wavs)
    stats['n_fonts'] = stats['n_wavs'] / stats['n_mids']

    time_tunes = []

    stats['time_tunes'] = OrderedDict()
    stats['notes'] = OrderedDict()
    stats['n_total_notes'] = np.zeros(128) + MIN_FLOAT
    stats['time_total_notes'] = np.zeros(128)

    polyphony = [0] * 128

    # Count totals by iterating over midi files and their notes.
    for f in mids:
        m = pretty_midi.PrettyMIDI(f)
        polyphony = update_poly_stats(polyphony, m)
        time_tunes.append(m.get_end_time())

        for i in m.instruments:
            if not i.is_drum:
                for n in i.notes:
                    stats['time_total_notes'][n.pitch] += n.end - n.start
                    stats['n_total_notes'][n.pitch] += 1

    # Calculate tune duration stats.
    stats['time_tunes']['total'] = units_format(sum(time_tunes))
    stats['time_tunes']['avg'] = units_format(np.mean(time_tunes))
    stats['time_tunes']['max'] = units_format(max(time_tunes))
    stats['time_tunes']['min'] = units_format(min(time_tunes))

    # Calculate additional stats.
    stats['time_avg_notes'] = stats['time_total_notes'] / \
                              stats['n_total_notes']
    stats['n_total_notes'] = stats['n_total_notes'].astype(int)
    stats['notes']['time_avg'] = np.mean(stats['time_avg_notes'])
    stats['notes']['n_total'] = np.sum(stats['n_total_notes'])
    stats['n_avg_notes'] = stats['n_total_notes'] / stats['n_mids']

    # Save polyphony stats.
    stats['polyphony_degree'] = polyphony

    return stats


def arrays_to_lists(d):
    '''Converts numpy arrays to python lists.'''

    for key in d.keys():
        if type(d[key]) is np.ndarray:
            d[key] = d[key].tolist()
    return d


def dict_as_json_string(stats):
    '''Converts dictionary object into json string.'''

    # Enable JSON serializability by converting arrays to lists
    stats = arrays_to_lists(stats)
    # Limit precision of float numbers in JSON representation
    json.encoder.FLOAT_REPR = lambda o: format(o, '.2f')

    # Hot-Fix Python 3 json serializability of numpy integer
    # https://stackoverflow.com/revisions/50577730/2
    def default(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    stats_json = json.dumps(
        stats, separators=(',', ':'), indent=4, default=default)
    refined = re.sub(r'\n\s+([0-9]+\.?[0-9]*|\])', r' \1', stats_json)
    return refined


def main():
    args = get_arguments()
    stats = get_stats(args.data_dir)
    stats_json_nice = dict_as_json_string(stats)
    print(stats_json_nice)


if __name__ == '__main__':
    main()
