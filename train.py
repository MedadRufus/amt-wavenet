'''Training script for the AMT WaveNet model.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from utils import Trainer, save_run_config, flush_n_close
from utils import roll_subsample
from utils import calc_stats, calc_metrics, metrics_empty_dict
from utils import write_metrics, write_images, write_audio

from wavenet import WaveNetModel, optimizer_factory, HKEY
from readers import WavMidReader


BATCH_SIZE = 20
DATA_DIRECTORY_TRAIN = './data/sanitycheck'
DATA_DIRECTORY_VALID = './data/sanitycheck'
LOGDIR_ROOT = './logs'
MODEL_PARAMS = './model_params.json'
TRAINING_PARAMS = './training_params.json'
RUNTIME_SWITCHES = './runtime_switches.json'
STARTED_DATESTRING = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())
SAMPLE_SIZE = 5000
MAX_TO_KEEP = 500
VELOCITY = False
THRESHOLD = 0.5


def get_arguments():
    def _str_to_bool(s):
        '''Convert string to bool (in argparse context).'''
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet for Transcription '
                                    '- training')

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many batch samples to process at once. '
                        'Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_dir_train', type=str,
                        default=DATA_DIRECTORY_TRAIN, help='The directory '
                        'containing the training data files. '
                        'Default: ' + DATA_DIRECTORY_TRAIN + '.')
    parser.add_argument('--data_dir_valid', type=str,
                        default=DATA_DIRECTORY_VALID, help='The directory '
                        'containing the validation data files. '
                        'Default: ' + DATA_DIRECTORY_VALID + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')

    parser.add_argument('--model_params', type=str, default=MODEL_PARAMS,
                        help='JSON file with the architecture hyperparameters. '
                        'Default: ' + MODEL_PARAMS + '.')
    parser.add_argument('--training_params', type=str, default=TRAINING_PARAMS,
                        help='JSON file with some training hyperparameters. '
                        'Default: ' + TRAINING_PARAMS + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. '
                        'Default: adam.')

    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be '
                        'kept alive. '
                        'Default: ' + str(MAX_TO_KEEP) + '.')
    parser.add_argument('--velocity', type=_str_to_bool, default=VELOCITY,
                        help='Whether to train to estimate velocity of '
                        'present notes. '
                        'Default: ' + str(VELOCITY) + '.')
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                        help='Threshold for post-processing. '
                        'Default: ' + str(THRESHOLD) + '.')

    return parser.parse_args()


def save(saver, sess, logdir, step):
    logdir = os.path.join(logdir, 'ckpt')
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end='')
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    logdir = os.path.join(logdir, 'ckpt')
    print('Trying to restore saved checkpoints from {} ...'.format(logdir),
          end='')

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print('  Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print('  Global step was: {}'.format(global_step))
        print('  Restoring...', end='')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(' Done.')
        return global_step
    else:
        print(' No checkpoint found.')
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    '''Validate and arrange directory related arguments.'''

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError('--logdir and --logdir_root cannot be '
                         'specified at the same time.')

    if args.logdir and args.restore_from:
        raise ValueError(
            '--logdir and --restore_from cannot be specified at the same '
            'time. This is to keep your previous model from unexpected '
            'overwrites.\n'
            'Use --logdir_root to specify the root of the directory which '
            'will be automatically created with current date and time, or use '
            'only --logdir to just continue the training from the last '
            'checkpoint.')

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are mutually exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def cumulateBatch(D, L, bD, bL):
    '''Appends data/label matrices to batch tensors.'''

    decrement = True

    try:
        bD = np.append(bD, D, axis=0) # batch-Data
        bL = np.append(bL, L, axis=0) # batch-Labels
    except ValueError:
        print('ValueError -> ignore |remaining samples of tune| < batch_size')
        print('D {} bD {} L {} bL {}'.format(
            D.shape, bD.shape, L.shape, bL.shape))
        decrement = False

    return bD, bL, decrement


def main():
    args = get_arguments()

    with open(args.model_params, 'r') as f:
        model_params = json.load(f)

    with open(args.training_params, 'r') as f:
        train_params = json.load(f)

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print('Some arguments are wrong:')
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    receptive_field = WaveNetModel.calculate_receptive_field(
        model_params['filter_width'],
        model_params['dilations'],
        model_params['initial_filter_width'])
    # Save arguments and model params into file
    save_run_config(args, receptive_field, STARTED_DATESTRING, logdir)

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Create data loader.
    with tf.name_scope('create_inputs'):
        reader = WavMidReader(data_dir=args.data_dir_train,
                              coord=coord,
                              audio_sample_rate=model_params['audio_sr'],
                              receptive_field=receptive_field,
                              velocity=args.velocity,
                              sample_size=args.sample_size,
                              queues_size=(10, 10*args.batch_size))
        data_batch = reader.dequeue(args.batch_size)

    # Create model.
    net = WaveNetModel(
        batch_size=args.batch_size,
        dilations=model_params['dilations'],
        filter_width=model_params['filter_width'],
        residual_channels=model_params['residual_channels'],
        dilation_channels=model_params['dilation_channels'],
        skip_channels=model_params['skip_channels'],
        output_channels=model_params['output_channels'],
        use_biases=model_params['use_biases'],
        initial_filter_width=model_params['initial_filter_width'])

    input_data = tf.placeholder(dtype=tf.float32,
                                shape=(args.batch_size, None, 1))
    input_labels = tf.placeholder(dtype=tf.float32,
                                  shape=(args.batch_size, None,
                                         model_params['output_channels']))

    loss, probs = net.loss(input_data=input_data,
                           input_labels=input_labels,
                           pos_weight=train_params['pos_weight'],
                           l2_reg_str=train_params['l2_reg_str'])
    optimizer = optimizer_factory[args.optimizer](
                    learning_rate=train_params['learning_rate'],
                    momentum=train_params['momentum'])
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    histograms = tf.summary.merge_all(key=HKEY)

    # Separate summary ops for validation, since they are
    # calculated only once per evaluation cycle.
    with tf.name_scope('validation_summaries'):

        metric_summaries = metrics_empty_dict()
        metric_value = tf.placeholder(tf.float32)
        for name in metric_summaries.keys():
            metric_summaries[name] = tf.summary.scalar(name, metric_value)

        images_buffer = tf.placeholder(tf.string)
        images_batch = tf.stack(
            [tf.image.decode_png(images_buffer[0], channels=4),
             tf.image.decode_png(images_buffer[1], channels=4),
             tf.image.decode_png(images_buffer[2], channels=4)])
        images_summary = tf.summary.image('estim', images_batch)

        audio_data = tf.placeholder(tf.float32)
        audio_summary = tf.summary.audio('input', audio_data,
                                         model_params['audio_sr'])

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(),
                           max_to_keep=args.max_checkpoints)

    # Trainer for keeping best validation-performing model
    # and optional early stopping.
    trainer = Trainer(sess, logdir, train_params['early_stop_limit'], 0.999)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print('Something went wrong while restoring checkpoint. '
              'Training will be terminated to avoid accidentally '
              'overwriting the previous model.')
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)


    step = None
    last_saved_step = saved_global_step
    try:
        for step in range(saved_global_step + 1, train_params['num_steps']):
            waveform, pianoroll = sess.run([data_batch[0], data_batch[1]])
            feed_dict = {input_data : waveform, input_labels : pianoroll}
            # Reload switches from file on each step
            with open(RUNTIME_SWITCHES, 'r') as f:
                switch = json.load(f)

            start_time = time.time()
            if switch['store_meta'] and step % switch['store_every'] == 0:
                # Slow run that stores extra information for debugging.
                print('Storing metadata')
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                summary, loss_value, _ = sess.run(
                    [summaries, loss, optim],
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata)
                writer.add_summary(summary, step)
                writer.add_run_metadata(run_metadata,
                                        'step_{:04d}'.format(step))
                tl = timeline.Timeline(run_metadata.step_stats)
                timeline_path = os.path.join(logdir, 'timeline.trace')
                with open(timeline_path, 'w') as f:
                    f.write(tl.generate_chrome_trace_format(show_memory=True))
            else:
                summary, loss_value, _ = sess.run([summaries, loss, optim],
                                                  feed_dict=feed_dict)
                writer.add_summary(summary, step)

            duration = time.time() - start_time
            print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                  .format(step, loss_value, duration))

            if step % switch['checkpoint_every'] == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

            # Evaluate model performance on validation data
            if step % switch['evaluate_every'] == 0:
                if switch['histograms']:
                    hist_summary = sess.run(histograms)
                    writer.add_summary(hist_summary, step)
                print('evaluating...')
                stats = 0, 0, 0, 0, 0, 0
                est = np.empty([0, model_params['output_channels']])
                ref = np.empty([0, model_params['output_channels']])

                b_data, b_labels, b_cntr = (
                    np.empty((0, args.sample_size + receptive_field - 1, 1)),
                    np.empty((0, model_params['output_channels'])),
                    args.batch_size)

                # if (batch_size * sample_size > valid_data) single_pass() again
                while est.size == 0: # and ref.size == 0 and sum(stats) == 0 ...

                    for data, labels in reader.single_pass(
                        sess, args.data_dir_valid):

                        # cumulate batch
                        if b_cntr > 1:
                            b_data, b_labels, decr = cumulateBatch(
                                data, labels, b_data, b_labels)
                            b_cntr -= decr
                            continue
                        elif args.batch_size > 1:
                            b_data, b_labels, decr = cumulateBatch(
                                data, labels, b_data, b_labels)
                            if not decr:
                                continue
                            data = b_data
                            labels = b_labels
                            # reset batch cumulation variables
                            b_data, b_labels, b_cntr = (
                                np.empty((
                                    0, args.sample_size + receptive_field - 1, 1
                                )),
                                np.empty((0, model_params['output_channels'])),
                                args.batch_size)

                        predictions = sess.run(
                            probs, feed_dict={input_data : data})
                        # Aggregate sums for metrics calculation
                        stats_chunk = calc_stats(
                            predictions, labels, args.threshold)
                        stats = tuple([sum(x) for x in zip(stats, stats_chunk)])
                        est = np.append(est, predictions, axis=0)
                        ref = np.append(ref, labels, axis=0)

                metrics = calc_metrics(None, None, None, stats=stats)
                write_metrics(metrics, metric_summaries, metric_value,
                              writer, step, sess)
                trainer.check(metrics['f1_measure'])

                # Render evaluation results
                if switch['log_image'] or switch['log_sound']:
                    sub_fac = int(model_params['audio_sr']/switch['midi_sr'])
                    est = roll_subsample(est.T, sub_fac)
                    ref = roll_subsample(ref.T, sub_fac)
                if switch['log_image']:
                    write_images(est, ref, switch['midi_sr'], args.threshold,
                                 (8, 6), images_summary, images_buffer,
                                 writer, step, sess)
                if switch['log_sound']:
                    write_audio(est, ref, switch['midi_sr'],
                                model_params['audio_sr'], 0.007,
                                audio_summary, audio_data,
                                writer, step, sess)

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)
        flush_n_close(writer, sess)


if __name__ == '__main__':
    main()
