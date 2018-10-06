'''Testing script for the WaveNet for Transcription.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(), os.pardir, os.pardir))
from utils import roll_subsample
from utils import calc_stats, calc_metrics
from utils import write_metrics, write_images, write_audio

from wavenet import WaveNetModel
from readers import WavMidReader

BATCH_SIZE = 1 # reasonable option for inference on longer audio pieces
DATA_DIRECTORY_TEST = './data/sanitycheck'
MODEL_PARAMS = './model_params.json'
RUNTIME_SWITCHES = './runtime_switches.json'
SAMPLE_SIZE = 112000
VELOCITY = False
THRESHOLD = 5e-1
PLOT_SCALE = 1e-0


def get_arguments():
    def _str_to_bool(s):
        '''Convert string to bool (in argparse context).'''
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet for Transcription '
                                    '- testing')
    parser.add_argument('--data_dir_test', type=str,
                        default=DATA_DIRECTORY_TEST,
                        help='The directory containing the testing data files. '
                        'Default: ' + DATA_DIRECTORY_TEST + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory with model checkpoint files. Also used '
                        'for evaluation output files.')
    parser.add_argument('--model_params', type=str, default=MODEL_PARAMS,
                        help='JSON file with the network parameters. Default: '
                        + MODEL_PARAMS + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Process test data this many audio samples at a '
                        'time. Should be a multiple of sub_fac (subsampling '
                        'factor) = audio_sr (default 16000) / midi_sr '
                        '(default 100). Largest batch_size that fits into 12GB '
                        'GPU RAM with default model size is 700*160. Therefore '
                        'Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--media', type=_str_to_bool, default=False,
                        help='Whether to store media (image & audio).'
                        'Default: False.')
    parser.add_argument('--velocity', type=_str_to_bool, default=VELOCITY,
                        help='Whether to train to estimate velocity of'
                        'present notes. '
                        'Default: ' + str(VELOCITY) + '.')
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                        help='Threshold for post-processing. '
                        'Default: ' + str(THRESHOLD) + '.')
    parser.add_argument('--plot_scale', type=float, default=PLOT_SCALE,
                        help='Scale for the size of image plots. '
                        'Default: ' + str(PLOT_SCALE) + '.')
    parser.add_argument('--plot_legend', type=_str_to_bool, default=True,
                        help='Whether to add legend to image plots. '
                        'Default: True.')
    return parser.parse_args()


def load(saver, sess, logdir):
    print('Trying to restore saved checkpoints from {} ...'.format(logdir),
          end='')

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print('  Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        print('  Restoring...', end='')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(' Done.')


def main():
    args = get_arguments()

    if (args.logdir is not None and os.path.isdir(args.logdir)):
        logdir = args.logdir
    else:
        print('Argument --logdir=\'{}\' is not (but should be) '
              'a path to valid directory.'.format(args.logdir))
        return

    with open(args.model_params, 'r') as f:
        model_params = json.load(f)
    with open(RUNTIME_SWITCHES, 'r') as f:
        switch = json.load(f)

    receptive_field = WaveNetModel.calculate_receptive_field(
        model_params['filter_width'],
        model_params['dilations'],
        model_params['initial_filter_width'])

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Create data loader.
    with tf.name_scope('create_inputs'):
        reader = WavMidReader(data_dir=args.data_dir_test,
                              coord=coord,
                              audio_sample_rate=model_params['audio_sr'],
                              receptive_field=receptive_field,
                              velocity=args.velocity,
                              sample_size=args.sample_size,
                              queues_size=(100, 100*BATCH_SIZE))

    # Create model.
    net = WaveNetModel(
        batch_size=BATCH_SIZE,
        dilations=model_params['dilations'],
        filter_width=model_params['filter_width'],
        residual_channels=model_params['residual_channels'],
        dilation_channels=model_params['dilation_channels'],
        skip_channels=model_params['skip_channels'],
        output_channels=model_params['output_channels'],
        use_biases=model_params['use_biases'],
        initial_filter_width=model_params['initial_filter_width'])

    input_data = tf.placeholder(dtype=tf.float32,
                                shape=(BATCH_SIZE, None, 1))
    input_labels = tf.placeholder(dtype=tf.float32,
                                  shape=(BATCH_SIZE, None,
                                         model_params['output_channels']))

    _, probs = net.loss(input_data=input_data,
                        input_labels=input_labels,
                        pos_weight=1.0,
                        l2_reg_str=None)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables())

    try:
        load(saver, sess, logdir)

    except:
        print('Something went wrong while restoring checkpoint.')
        raise

    try:
        stats = 0, 0, 0, 0, 0, 0
        est = np.empty([model_params['output_channels'], 0])
        ref = np.empty([model_params['output_channels'], 0])
        sub_fac = int(model_params['audio_sr']/switch['midi_sr'])
        for data, labels in reader.single_pass(sess,
                                               args.data_dir_test):

            predictions = sess.run(probs, feed_dict={input_data : data})
            # Aggregate sums for metrics calculation
            stats_chunk = calc_stats(predictions, labels, args.threshold)
            stats = tuple([sum(x) for x in zip(stats, stats_chunk)])
            est = np.append(est, roll_subsample(predictions.T, sub_fac), axis=1)
            ref = np.append(ref, roll_subsample(labels.T, sub_fac, b=True),
                            axis=1)

        metrics = calc_metrics(None, None, None, stats=stats)
        write_metrics(metrics, None, None, None, None, None, logdir=logdir)

        # Save subsampled data for further arbitrary evaluation
        np.save(logdir+'/est.npy', est)
        np.save(logdir+'/ref.npy', ref)

        # Render evaluation results
        figsize=(int(args.plot_scale*est.shape[1]/switch['midi_sr']),
                 int(args.plot_scale*model_params['output_channels']/12))
        if args.media:
            write_images(est, ref, switch['midi_sr'],
                         args.threshold, figsize,
                         None, None, None, 0, None,
                         noterange=(21, 109),
                         legend=args.plot_legend,
                         logdir=logdir)
            write_audio(est, ref, switch['midi_sr'],
                        model_params['audio_sr'], 0.007,
                        None, None, None, 0, None, logdir=logdir)

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        coord.request_stop()


if __name__ == '__main__':
    main()
