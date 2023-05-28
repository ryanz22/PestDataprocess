import io

import pytest
from pathlib import Path

from dataprocess.util.presentation import load_meta, parse_meta, Meta

# https://medium.com/testcult/intro-to-test-framework-pytest-5b1ce4d011ae
# https://medium.com/beyn-technology/hands-on-start-testing-with-pytest-1ef39e59176a
# https://medium.com/ideas-at-igenius/fixtures-and-parameters-testing-code-with-pytest-d8603abb390a
# https://betterprogramming.pub/understand-5-scopes-of-pytest-fixtures-1b607b5c19ed


@pytest.mark.parametrize(
    "yaml_str, parent, expected",
    [
        (
            """
title: A mix of two grasshoppers in the same family without drone noise
description: kernel size 8, stride 4
src_cnt: 2
has_noise: false
src_mappings:
  mix: mix.wav
  s1: s1.wav
  s1_desc: grasshopper
  s2: s2.wav
  s2_desc: another grasshopper
  est_s1: est_s1.wav
  est_s2: est_s2.wav
    """,
            Path("/test"),
            Meta(
                title="A mix of two grasshoppers in the same family without drone noise",
                description="kernel size 8, stride 4",
                src_cnt=2,
                has_noise=False,
                mix_fn=Path("/test") / "mix.wav",
                s1_fn=Path("/test") / "s1.wav",
                s1_desc="grasshopper",
                s2_fn=Path("/test") / "s2.wav",
                s2_desc="another grasshopper",
                s3_fn=None,
                s3_desc=None,
                est_s1_fn=Path("/test") / "est_s1.wav",
                est_s2_fn=Path("/test") / "est_s2.wav",
                est_s3_fn=None,
                noise_fn=None,
            ),
        ),
        (
            """
title: A mix of grasshopper, bird and cricket with drone noise
description: kernel size 8, stride 4
src_cnt: 3
has_noise: true
src_mappings:
  mix: mix.wav
  s1: s1.wav
  s1_desc: grasshopper
  s2: s2.wav
  s2_desc: bird
  s3: s3.wav
  s3_desc: cricket
  est_s1: est_s1.wav
  est_s2: est_s2.wav
  est_s3: est_s3.wav
  noise: noise.wav
    """,
            Path("/test"),
            Meta(
                title="A mix of grasshopper, bird and cricket with drone noise",
                description="kernel size 8, stride 4",
                src_cnt=3,
                has_noise=True,
                mix_fn=Path("/test") / "mix.wav",
                s1_fn=Path("/test") / "s1.wav",
                s1_desc="grasshopper",
                s2_fn=Path("/test") / "s2.wav",
                s2_desc="bird",
                s3_fn=Path("/test") / "s3.wav",
                s3_desc="cricket",
                est_s1_fn=Path("/test") / "est_s1.wav",
                est_s2_fn=Path("/test") / "est_s2.wav",
                est_s3_fn=Path("/test") / "est_s3.wav",
                noise_fn=Path("/test") / "noise.wav",
            ),
        ),
    ],
)
def test_load_meta(yaml_str, parent, expected):
    assert parse_meta(yaml_str, parent, validate=False).unwrap() == expected


def is_odd(num: int) -> bool:
    return num % 2 != 0


@pytest.fixture(params=range(1, 11, 2))
def odd(request):
    return request.param


@pytest.fixture(params=range(0, 10, 2))
def even(request):
    return request.param


def test_sum_odd_even_returns_odd(odd, even):
    assert is_odd(odd + even)
