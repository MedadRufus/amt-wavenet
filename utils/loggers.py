import numpy as np
import librosa
import json

from .renderers import (plot_estim, plot_eval, plot_certainty,
                        roll2audio)


def save_content(content,
                 summary_op,
                 placeholder,
                 writer,
                 step,
                 sess):
    '''Saves summary data to TensorBoard.'''

    feed_dict = {placeholder : content}
    summary = sess.run(summary_op,
                       feed_dict=feed_dict)
    writer.add_summary(summary, step)


def write_metrics(metrics,
                  summary_ops,
                  placeholder,
                  writer,
                  step,
                  sess,
                  logdir=None):
    '''Writes metrics to TensorBoard and/or disk.'''

    if logdir:
        with open(logdir+'/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

    if sess:
        for name, value in metrics.items():
            save_content(value, summary_ops[name],
                         placeholder, writer, step, sess)


def write_images(est, ref, sr, thr, figsize,
                 summary_op, placeholder,
                 writer, step, sess,
                 noterange=(21, 109), legend=False, logdir=None):
    '''Renders evaluation plots from estimated and reference
    piano rolls and writes images as summary to TensorBoard and/or disk.
    '''

    buff_estim = plot_estim(
        est, sr, colorbar=True, noterange=noterange, figsize=figsize)
    buff_cert = plot_certainty(
        est, ref, sr, legend=legend, noterange=noterange, figsize=figsize)
    buff_eval = plot_eval(
        est, ref, sr, thr, legend=legend, noterange=noterange, figsize=figsize)

    if logdir:
        open(logdir+'/estim-'+str(step)+'.png', 'wb'
            ).write(buff_estim.getvalue())
        open(logdir+'/cert-'+str(step)+'.png', 'wb'
            ).write(buff_cert.getvalue())
        open(logdir+'/eval-'+str(step)+'.png', 'wb'
            ).write(buff_eval.getvalue())

    if sess:
        images_buffer = [buff_estim.getvalue(),
                         buff_eval.getvalue(),
                         buff_cert.getvalue()]

        save_content(images_buffer, summary_op, placeholder,
                     writer, step, sess)


def write_audio(est, ref, midi_sr, audio_sr, thr,
                summary_op, placeholder,
                writer, step, sess, logdir=None):
    '''Write audio summary to TensorBoard and/or disk.'''

    wav_predic = roll2audio(est, midi_sr, audio_sr, thr=thr)
    wav_labels = roll2audio(ref, midi_sr, audio_sr, thr=thr)

    if wav_predic.size != wav_labels.size:
        diff = np.abs(wav_predic.size - wav_labels.size)
        if wav_predic.size < wav_labels.size:
            wav_predic = np.pad(wav_predic, (0, diff), 'constant')
        elif wav_predic.size > wav_labels.size:
            wav_labels = np.pad(wav_labels, (0, diff), 'constant')

    if logdir:
        librosa.output.write_wav(
            logdir+'/transc-'+str(step)+'.wav', wav_predic, audio_sr)
        librosa.output.write_wav(
            logdir+'/origin-'+str(step)+'.wav', wav_labels, audio_sr)

    if wav_predic.size > 0 and sess:
        wav_merged = np.array([wav_predic.astype(np.float32),
                               wav_labels.astype(np.float32)])

        save_content(wav_merged, summary_op, placeholder,
                     writer, step, sess)


def tb_write_histograms(summary_ops, writer, step):
    '''Write histogram summaries of all trainable parameters to TensorBoard'''

    for his_summ in summary_ops:
        writer.add_summary(his_summ.eval(), step)
