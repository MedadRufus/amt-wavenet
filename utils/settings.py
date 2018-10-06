import inspect
import shutil
import os
import tensorflow as tf


class Trainer(object):
    '''Manages checkpointing on improvements, early stopping
    and restoring session from last checkpoint.
    '''

    def __init__(self, sess, logdir, limit, stopbound):
        self.sess = sess
        self.logdir = logdir
        self.early_stop_limit = limit
        self.stopbound = stopbound

        self.saver = tf.train.Saver()
        self.metric_best = 0
        self.checks_unimproved = 0

        if not os.path.exists(logdir+'/best'):
            os.makedirs(logdir+'/best')
        self.FILENAME = logdir+'/best/best.ckpt'

    def check(self, metric):
        '''Keeps track of performance metric, saves model on
        improvements and terminates training upon stagnation or
        satisfaction.
        '''

        if self.metric_best < metric:
            self.metric_best = metric
            self.saver.save(
                self.sess,
                self.FILENAME)
            self.checks_unimproved = 0
        elif self.early_stop_limit:
            self.checks_unimproved = self.checks_unimproved + 1
            if (self.checks_unimproved == self.early_stop_limit):
                print('Trainer stops training after {:} checks without '
                      'improvemet of best metric value {:.4f}'.format(
                        self.checks_unimproved, self.metric_best))
                raise KeyboardInterrupt()

        if metric > self.stopbound:
            print('Trainer stops training since metric crossed stop boundary '
                  '{} with value {}'.format(self.stopbound, metric))
            raise KeyboardInterrupt()


def flush_n_close(writer, session):
    '''Flush remaining event files to disk, close writer
    and session.
    '''

    writer.flush()
    writer.close()
    session.close()


def save_run_config(args, recf, timestamp, logdir, mode='a'):
    '''Saves dictionary with config values into log directory.'''

    import json
    from collections import OrderedDict
    _EL = '\n\n' # empty line

    # Add script arguments
    cfg = 'args:' + _EL + json.dumps(args.__dict__, indent=4, sort_keys=True)

    # Add training parameters
    with open(args.training_params, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
        for metakey in [k for k in data.keys() if k.startswith('_')]:
            data.pop(metakey)
        cfg += _EL + 'training_params:' + _EL + json.dumps(data, indent=4)

    # Add model parameters
    with open(args.model_params, 'r') as f:
        cfg += _EL + 'model_params:' + _EL + str(f.read())

    cfg += _EL + 'receptive_field: ' + str(recf)

    # Save run configuration
    fname = logdir + '/setup-' + timestamp + '.log'

    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    f = open(fname, mode)
    f.write(cfg)
    f.close()

    print('Run configuration saved into file {}'.format(fname))
