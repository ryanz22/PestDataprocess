import io

import pytest
from pathlib import Path

from dataprocess.sound.audio_id_data import get_gh_split_lists


@pytest.mark.parametrize(
    "root, split, expected",
    [(Path("data/sound/unit_test/gh-raw-ds-mini"), (0.7, 0.2, 0.1), (34, 9, 9))],
)
def test_get_gh_split_lists(root, split, expected):
    train, dev, test = get_gh_split_lists(root, split)
    assert (len(train), len(dev), len(test)) == expected
