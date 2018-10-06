from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import random
import re
import threading

import librosa
import pretty_midi
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(), os.pardir))
from utils import find_files, roll_encode, roll_decode, get_roll_index


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def load_piece_data(directory,
                    audio_sr,
                    velocity,
                    fac,
                    valid=True):
    '''Loader that reads tune from directory and yields audio
    waveform and encoded piano roll as tuple of 3 arrays:
    (W, T, I). If more audio files represent single midi file,
    one is chosen randomly.
    '''

    midi_files = find_files(directory, '*.mid')
    randomized_midi_files = randomize_files(midi_files)

    for midi_filename in randomized_midi_files if valid else midi_files:
        # load piano roll from midi file
        proll = pretty_midi.PrettyMIDI(
            midi_filename).get_piano_roll(fs=int(audio_sr/fac))
        proll /= 127 # velocity to <0;1>
        if not velocity:
            proll[proll > 0] = 1
        # encode piano roll
        table, indices = roll_encode(proll, fac)
        # add 0-roll if not present (we will need it later for padding)
        if get_roll_index(table, np.zeros(128)).shape[0] == 0:
            table = np.concatenate((table, np.zeros(shape=(1, 128))))
        # get respective audio file names and choose 1 randomly
        base = midi_filename.rsplit('/', 1)[-1]
        base = re.sub(r'(.*)%s$' % re.escape('.mid'), r'\1', base)
        audio_files = find_files(directory, base+'*.wav')
        if not audio_files:
            raise ValueError('No files found for \'{}\'.'.format(base+'*.wav'))
        audio_filename = random.choice(audio_files)
        # load audio waveform
        audio, _ = librosa.load(audio_filename, sr=audio_sr, mono=True)
        yield audio, table, indices


def sequence_samples(audio, table, indices, reader):
    '''Generator that yields batch samples as a tuple
    of numpy arrays (wave, roll) with shapes:
        wave.shape = (sample_size + receptive_field - 1, 1)
        roll.shape = (sample_size, 128)
    where sample_size is length of slice to which piece
    is cut. Last slice of a tune may have shape with
    length < sample_size.
    '''

    left = np.ceil((reader.receptive_field - 1) / 2).astype(int)
    right = np.floor((reader.receptive_field - 1) / 2).astype(int)
    # Ensure len(audio) == len(indices)
    if (audio.shape[0] < indices.shape[0]):
        # Cut piano roll down to length of audio sequence
        indices = indices[:audio.shape[0]]
    else:
        # Pad piano roll up to length of audio sequence, since this is
        # usually longer due to sustain of last notes
        indices = np.pad(indices,
                        [0, audio.shape[0] - indices.shape[0]],
                        'constant',
                        constant_values=get_roll_index(table, np.zeros(128))[0])
    # Pad audio sequence from left and right to provide context
    # to each estimate, receptive field is therefore centered
    # to time sample being calculated
    audio = np.pad(audio,
                   [left, right],
                   'constant').reshape(-1, 1)

    if reader.sample_size:
        # Cut tune into sequences of size sample_size +
        # receptive_field - 1 with overlap = receptive_field - 1
        while len(audio) > reader.receptive_field:
            wave = audio[:(left + reader.sample_size + right), :]
            roll = roll_decode(table, indices[:reader.sample_size])
            yield wave, roll

            audio = audio[reader.sample_size:, :]
            indices = indices[reader.sample_size:]
    else:
        yield audio, roll_decode(table, indices)


class WavMidReader(object):
    '''Generic background music data reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.
    '''

    def __init__(self,
                 data_dir,
                 coord,
                 audio_sample_rate,
                 receptive_field,
                 velocity,
                 sample_size,
                 queues_size,
                 compress_fac=10):
        self.data_dir = data_dir
        self.audio_sample_rate = audio_sample_rate
        self.compress_fac = compress_fac
        self.coord = coord
        self.receptive_field = receptive_field
        self.velocity = velocity
        self.sample_size = sample_size
        self.threads = []

        # Init queues and placeholders.
        self.queues = {'tune': {}, 'batch': {}}
        self.audio_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.table_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,
                                                                         128))
        self.indices_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.queues['tune']['Q'] = tf.FIFOQueue(queues_size[0],
                                                ['float32', 'float32', 'int32'])
        self.queues['tune']['enQ'] = self.queues['tune']['Q'].enqueue(
            [self.audio_placeholder,
             self.table_placeholder,
             self.indices_placeholder])
        self.wave_placeholder = tf.placeholder(dtype=tf.float32,
                                               shape=(None, 1))
        self.roll_placeholder = tf.placeholder(dtype=tf.float32,
                                               shape=(None, 128))
        self.queues['batch']['Q'] = tf.PaddingFIFOQueue(
            queues_size[1], ['float32', 'float32'],
            shapes=[(None, 1), (None, 128)])
        self.queues['batch']['enQ'] = self.queues['batch']['Q'].enqueue(
            [self.wave_placeholder, self.roll_placeholder])

        self.file_counter = tf.Variable(0, trainable=True)
        self.increment_file_counter = tf.assign(
            self.file_counter, self.file_counter+1)

        files = find_files(data_dir, '*.mid')
        if not files:
            raise ValueError('No midi files found in \'{}\'.'.format(data_dir))

    def dequeue(self, num_elements):
        output = self.queues['batch']['Q'].dequeue_many(num_elements)
        return output

    def thread_loader(self, sess):
        stop = False
        # Count tune data files
        n_midi_files = len(find_files(self.data_dir, '*.mid'))
        if n_midi_files == 0:
            raise ValueError('No files found for \'{}\'.'.format(
                directory+'/*.mid'))
        one_percent = int(np.ceil(n_midi_files/100))
        print('files length: {}'.format(n_midi_files))
        # Go through the dataset repeatedly until stopped
        while not stop:
            # Randomly iterate over files and fetch tune data
            file_iterator = load_piece_data(self.data_dir,
                                            self.audio_sample_rate,
                                            self.velocity,
                                            self.compress_fac)
            for audio, table, indices in file_iterator:
                sess.run(self.queues['tune']['enQ'],
                         feed_dict={self.audio_placeholder: audio,
                                    self.table_placeholder: table,
                                    self.indices_placeholder: indices})
                # Track and report progress
                sess.run(self.increment_file_counter)
                file_counter = sess.run(self.file_counter)
                if file_counter % one_percent == 0:
                    print('Training progress: {:.02f} epochs '
                          '(file {} of {})'.format(file_counter/n_midi_files,
                                                   file_counter, n_midi_files))
                if self.coord.should_stop():
                    stop = True
                    break

    def thread_generator(self, sess):
        stop = False
        # Go through the dataset repeatedly until stopped
        while not stop:
            # Dequeue tune data
            audio, table, indices = sess.run(self.queues['tune']['Q'].dequeue())
            # Fetch samples from the tune
            sample_iterator = sequence_samples(audio, table, indices, self)
            for wave, roll in sample_iterator:
                sess.run(self.queues['batch']['enQ'],
                         feed_dict={self.wave_placeholder: wave,
                                    self.roll_placeholder: roll})
                if self.coord.should_stop():
                    stop = True
                    break

    def single_pass(self, sess, data_dir):
        for audio, table, indices in load_piece_data(data_dir,
                                                     self.audio_sample_rate,
                                                     self.velocity,
                                                     self.compress_fac,
                                                     valid=False):
            if self.coord.should_stop():
                break
            for wave, roll in sequence_samples(audio,
                                               table,
                                               indices,
                                               self):
                if self.coord.should_stop():
                    break
                wave = np.expand_dims(wave, axis=0)
                yield wave, roll

    def start_threads(self, sess, n_threads=1):
        def _add_daemon_thread(reader, thread_func, sess):
            thread = threading.Thread(target=thread_func, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            reader.threads.append(thread)

        # Single loader will suffice to possibly multiple generators
        _add_daemon_thread(self, self.thread_loader, sess)
        for _ in range(n_threads):
            _add_daemon_thread(self, self.thread_generator, sess)

        for thread in self.threads:
            thread.start()
        return self.threads
