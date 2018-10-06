# Sanity check

Dataset for use in the simplest test case: overfit WaveNet to the middle C synthesized with simple sine wave.

## Data origin

1. in `bash`:
```
wget -O C.mid https://upload.wikimedia.org/wikipedia/commons/e/e2/Middle_C.mid
```
2. in `python`:
```
>>> import pretty_midi, librosa
>>> midi = pretty_midi.PrettyMIDI('C.mid')
>>> audio = midi.synthesize(fs=16000)
>>> librosa.output.write_wav('C.wav', audio, 16000)
```
