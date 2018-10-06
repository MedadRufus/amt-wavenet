import numpy as np


def thresh(values, threshold, retain):
    if retain:
        values = np.copy(values)
    values[values<threshold] = 0
    values[values>=threshold] = 1
    return values


def roll_encode(m, fac):
    '''Encodes piano roll into table of unique slices and
    vector of indices into this table.

    Thanks for inspiration: https://stackoverflow.com/a/18197790
    '''

    mt = m.T
    b = np.ascontiguousarray(mt).view(
        np.dtype((np.void, mt.dtype.itemsize * mt.shape[1])))
    _, idx, indices = np.unique(b,
                                return_index=True,
                                return_inverse=True)
    tab = m[:, idx].T

    return tab, np.array([[i] * fac for i in indices]).reshape(-1)


def roll_decode(tab, indices):
    '''Decodes piano roll by substitution of `indices` with
    their respective values in `tab`. Result is stretched
    by factor `fac`.
    '''

    roll = [tab[i] for i in indices]
    return np.array(roll).reshape(-1, 128)


def get_roll_index(m, r):
    '''Returns index of roll `r` from table `m` of unique rolls.'''

    return np.array(
        np.where(
            np.all(m==r, axis=1)
        )
    ).reshape(-1)


def roll_subsample(a, n, b=False):
    '''Subsample array `a` by factor of `n` samples.'''

    def _subsample(a, n):
        return [np.mean(a[i: i+n]) for i in range(0, len(a), n)]

    s = np.apply_along_axis(_subsample, 1, a, n)

    if b: # binarize outputs
        s[s<0.5] = 0.
        s[s>=0.5] = 1.

    return s
