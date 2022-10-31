import numpy as np
from dataprocess.util.data_process import fill_fix_len


def test_fill_fix_len():
    sr = 10
    dura1 = 5.0
    a = np.arange(0, int(sr * dura1), 1)
    dura2 = 7.0
    b = fill_fix_len(a, sr, dura2)
    assert len(b) == int(sr * dura2)

    c = fill_fix_len(a, sr, dura1)
    assert len(c) == int(sr * dura1)
