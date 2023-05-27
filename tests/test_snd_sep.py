import io

import pytest
from pathlib import Path

from dataprocess.util.presentation import load_meta, parse_meta, Meta

# https://medium.com/testcult/intro-to-test-framework-pytest-5b1ce4d011ae


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
        )
    ],
)
def test_load_meta(yaml_str, parent, expected):
    assert parse_meta(yaml_str, parent, validate=False).unwrap() == expected
