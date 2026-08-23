import numpy as np
import pytest

from dataprocess.sound.preprocess import mix, normalize, resample, to_mono


def sine(sr: int, freq: float, seconds: float) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_resample_round_trip_length():
    sr_in, sr_out = 44100, 22050
    y = sine(sr_in, 440.0, 2.0)
    y2, sr = resample(y, sr_in, sr_out)
    assert sr == sr_out
    assert len(y2) == pytest.approx(len(y) * sr_out / sr_in, rel=0.01)


def test_to_mono_stereo():
    sr = 22050
    stereo = np.stack([sine(sr, 440.0, 1.0), sine(sr, 880.0, 1.0)])
    mono, sr2 = to_mono(stereo, sr)
    assert sr2 == sr
    assert mono.ndim == 1


def test_normalize_stereo_input_returns_mono():
    sr = 44100
    stereo = np.stack([sine(sr, 440.0, 1.0), sine(sr, 440.0, 1.0)])
    out, out_sr = normalize(stereo, sr, tsr=22050)
    assert out_sr == 22050
    assert out.ndim == 1


def test_mix_shorter_mode_length():
    sr = 22050
    a = (sine(sr, 440.0, 3.0), sr)
    b_path_long = sine(sr, 880.0, 5.0)
    mixed, sr2 = mix(a, _tmp_wav(b_path_long, sr), mode="shorter")
    assert sr2 == sr
    assert len(mixed) == int(3.0 * sr)


def test_mix_sample_rate_mismatch_raises():
    a = (sine(22050, 440.0, 1.0), 22050)
    b = _tmp_wav(sine(44100, 880.0, 1.0), 44100)
    with pytest.raises(ValueError):
        mix(a, b)


def test_cwt_does_not_mutate_input():
    from dataprocess.cwt.cwt2 import cwt2

    x = sine(22050, 440.0, 0.5).astype(np.float64)
    before = x.copy()
    cwt2(x)
    assert np.array_equal(x, before)


def _tmp_wav(y: np.ndarray, sr: int) -> str:
    import tempfile
    from pathlib import Path
    import soundfile as sf

    p = Path(tempfile.mkdtemp()) / "in.wav"
    sf.write(p, y, sr)
    return str(p)
