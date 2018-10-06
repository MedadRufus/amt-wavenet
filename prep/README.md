# Preprocessing
Utility scripts for manipulating datasets.

## Usage

See `--help` switch of respective script:
```
python csv2mid.py --help
python summ.py --help
python trim.py --help
```

## Examples
* `csv2mid.py` can be used to convert [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html) dataset labels in its [raw distribution](https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz) from `csv` to `mid`, as expected by [wavmid_reader.py](../readers/wavmid_reader.py)
* `summ.py` generates statistics of the dataset (of `mid`, `wav` pairs) contained in supplied directory
* `trim.py` can be used to cut specified range (in seconds) from each tune in the dataset, e.g. to replicate dataset setup used [here](https://labrosa.ee.columbia.edu/projects/piano/).
