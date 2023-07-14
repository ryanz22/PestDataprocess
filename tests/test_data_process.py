import numpy as np
from dataprocess.util.data_process import fill_fix_len, split_list
import pytest


def test_fill_fix_len():
    sr = 10
    dura1 = 5.0
    a = np.arange(0, int(sr * dura1), 1)
    dura2 = 7.0
    b = fill_fix_len(a, sr, dura2)
    assert len(b) == int(sr * dura2)

    c = fill_fix_len(a, sr, dura1)
    assert len(c) == int(sr * dura1)


@pytest.mark.parametrize(
    "list_len, factors, expected",
    [
        (10, (0.7, 0.2, 0.1), [7, 2, 1]),
        (10, (0.7, 0.4, 0.2), [7, 3, 0]),
        (1234, (0.7, 0.2, 0.1), [864, 247, 123]),
    ],
)
def test_split_list(list_len, factors, expected):
    l1, l2, l3 = split_list([x for x in range(list_len)], factors)
    assert [len(l1), len(l2), len(l3)] == expected
