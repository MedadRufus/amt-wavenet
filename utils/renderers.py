from __future__ import division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
import mir_eval
import mir_eval.display
import librosa.display
import pretty_midi
import io
import itertools
import numpy as np

from .piano_roll import thresh


def piano_roll2midi_events(piano_roll, fs, dynamic=False):
    '''Converts piano roll into list of midi events,
    which is a standard format for evaluation in MIR
    required also by `mir_eval.display.piano_roll`.
    '''

    times = np.empty([0,2])
    midis = np.empty([0])
    velocities = list()
    t_start = -1

    for note in np.unique(np.nonzero(piano_roll)[0]):
        for t in range(piano_roll.shape[1]):
            if (t_start == -1 and
                piano_roll[note][t] != 0 and
                t < piano_roll.shape[1]-1):
                t_start = t
            elif (t_start != -1 and
                  (piano_roll[note][t] == 0 or t == piano_roll.shape[1]-1)):
                times = np.append(times, [[t_start/fs, t/fs]], axis=0)
                midis = np.append(midis, [note], axis=0)
                if dynamic:
                    velocities.append(np.mean(piano_roll[note][t_start:t]))
                t_start = -1

    return (times, midis, velocities) if dynamic else (times, midis)


def roll2audio(piano_roll,
               midi_sr,
               audio_sr,
               thr=0.01,
               prg=-1,
               sf2=None):
    '''Converts piano roll to PrettyMIDI object and renders audio
    from this object using fluidsynth with default soundfont.
    '''

    roll = np.copy(piano_roll)
    roll[roll<thr] = 0

    times, midis, velocities = piano_roll2midi_events(
        roll, midi_sr, dynamic=True)

    track = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(prg)
    for t, m, v in zip(times, midis, velocities):
        note = pretty_midi.Note(velocity=int(v*127),
                                pitch=int(m),
                                start=t[0],
                                end=t[1])
        instr.notes.append(note)
    track.instruments.append(instr)

    return (track.synthesize(fs=audio_sr) if prg == -1 else
            track.fluidsynth(fs=audio_sr, sf2_path=sf2))


def plot_eval(predictions,
              labels,
              fs,
              thr,
              showfig=False,
              retain=False,
              legend=False,
              noterange=(21, 109),
              figsize=(4*4, 1*4)):
    '''Plots IR-style evaluation matrix by comparing
    thresholded labels with thresholded estimations.
    IR task Outcomes are visualized in colors.
    '''

    # Obtain thresholded labels and predictions
    p = thresh(predictions, thr, retain).astype(bool)
    l = thresh(labels, thr, retain).astype(bool)

    # true positives (1 AND 1)
    TP_times, TP_midis = piano_roll2midi_events(p * l, fs)
    # false negatives ((0 XOR 1) AND 1)
    FN_times, FN_midis = piano_roll2midi_events((p ^ l) * l, fs)
    # false positives ((1 XOR 0) AND 1)
    FP_times, FP_midis = piano_roll2midi_events((p ^ l) * p, fs)

    plt.ion()
    fig = plt.figure(figsize=figsize)
    mir_eval.display.piano_roll(TP_times,
                                midi=TP_midis,
                                label='True Positive',
                                facecolor=(0, 1, 0, 1),
                                linewidth=0)
    mir_eval.display.piano_roll(FN_times,
                                midi=FN_midis,
                                label='False Negative',
                                facecolor=(1, 0, 0, 1),
                                linewidth=0)
    mir_eval.display.piano_roll(FP_times,
                                midi=FP_midis,
                                label='False Positive',
                                facecolor=(0, 0, 1, 1),
                                linewidth=0)

    plt.grid(True, which='major')
    plt.grid(True, which='minor', alpha=0.25)
    if noterange:
        fig.get_axes()[0].set_ybound(lower=noterange[0],
                                     upper=noterange[1])
    if legend:
        plt.legend(mode='best')
    mir_eval.display.ticker_notes()
    loc = matplotlib.ticker.MultipleLocator(base=12.0)
    fig.get_axes()[0].yaxis.set_major_locator(loc)
    plt.tight_layout()

    buff = io.BytesIO()
    fig.savefig(buff, format='png')
    buff.seek(0)

    if showfig:
        plt.show(block=False)
    else:
        plt.close(fig)

    return buff


def plot_estim(roll,
               fs,
               showfig=False,
               colorbar=False,
               noterange=(21, 109),
               figsize=(4*4, 1*4),
               cmap='Greys'):
    '''Plots estimated matrix using `librosa.display.specshow`.'''

    plt.ion()
    fig = plt.figure(figsize=figsize)
    librosa.display.specshow(roll[noterange[0]:noterange[1], :],
                             sr=fs,
                             hop_length=1,
                             x_axis='time',
                             #y_axis='cqt_note', # FIXME: correct alignment
                             bins_per_octave = 12,
                             cmap=cmap)

    if colorbar:
        plt.colorbar()
    plt.tight_layout()

    buff = io.BytesIO()
    fig.savefig(buff, format='png')
    buff.seek(0)

    if showfig:
        plt.show(block=False)
    else:
        plt.close(fig)

    return buff


def plot_certainty(predictions,
                   labels,
                   fs,
                   thr=0.01,
                   showfig=False,
                   legend=False,
                   noterange=(21, 109),
                   figsize=(4*4, 1*4),
                   rgb=(1,0,0)):
    '''Plots real-valued estimations (certainty/velocity) against
    binary-valued estimations (only frames of notes ar drawn).
    '''

    # TODO: Fix this dependency with a PR to mir_eval
    '''diff --git a/mir_eval/display.py b/mir_eval/display.py
    index 02454d8..0af0992 100644
    --- a/mir_eval/display.py
    +++ b/mir_eval/display.py
    @@ -298,6 +298,10 @@ def labeled_intervals(intervals, labels, label_set=None,
             # Pop the label after the first time we see it, so we only get
             # one legend entry
             style.pop('label', None)
    +        # Treat the case where intervals have individual color labels
    +        if (type(style['facecolor']) == list and
    +            len(style['facecolor']) >= len(xvals[lab])):
    +            del style['facecolor'][:len(xvals[lab])]
         # Draw a line separating the new labels from pre-existing labels
         if label_set != ticks:
    '''

    p = np.copy(predictions)
    l = labels

    p[p<thr] = 0

    p_times, p_midis, p_colors = piano_roll2midi_events(
        p, fs, dynamic=True)
    l_times, l_midis = piano_roll2midi_events(l, fs)

    p_colors = [rgb + (p,) for p in p_colors]

    plt.ion()
    fig = plt.figure(figsize=figsize)
    mir_eval.display.piano_roll(p_times,
                                midi=p_midis,
                                label='Estimate',
                                linewidth=1,
                                edgecolor=rgb+(1,),
                                facecolor=p_colors)
    mir_eval.display.piano_roll(l_times,
                                midi=l_midis,
                                label='Reference',
                                linewidth=1,
                                edgecolor=(0, 0, 1, 1),
                                facecolor=(0, 0, 0, 0))

    plt.grid(True, which='major')
    plt.grid(True, which='minor', alpha=0.25)
    if noterange:
        fig.get_axes()[0].set_ybound(lower=noterange[0],
                                     upper=noterange[1])
    if legend:
        plt.legend(mode='best')
    mir_eval.display.ticker_notes()
    loc = matplotlib.ticker.MultipleLocator(base=12.0)
    fig.get_axes()[0].yaxis.set_major_locator(loc)
    plt.tight_layout()

    buff = io.BytesIO()
    fig.savefig(buff, format='png')
    buff.seek(0)

    if showfig:
        plt.show(block=False)
    else:
        plt.close(fig)

    return buff

# TODO: float labels float estimations (velocity-sensitive training)
