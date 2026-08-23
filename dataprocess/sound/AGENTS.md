# dataprocess/sound/

Core audio domain logic + vendored source-separation code.

## VENDORED NUSSL (read-only)

`nussl/` is a trimmed vendor copy of [nussl](https://github.com/nussl/nussl): only `core/audio_signal.py` (~2600 ln) + `core/masks/`. The pip dependency is intentionally commented out (pyproject.toml `#nussl = "^1.1.9"`). No local modifications detected (no `from dataprocess` imports inside). Never edit; if separation features are needed, extend wrappers outside `nussl/`.

## MODULES

| File | Role |
|------|------|
| `preprocess.py` | mono conversion, noisereduce denoising, resampling, sound-file info |
| `audio_augment.py` | augmentation via scaper (mix w/ backgrounds) + audiomentations |
| `audio_filter.py` / `filter_util.py` | filter class (shape-checked) + test utilities |
| `sep_data.py` (~357 ln) | builds source-separation datasets: mix.wav + s1..s3 (+noise) with per-sample YAML meta consumed by `util/presentation.Meta` |
| `audio_id_data.py` | audio-identification dataset prep |
| `plot_wav.py` | waveform/spectrogram/scalogram plotting |

## CONVENTIONS

- Processed outputs keep base name + suffix chain: `_mono_<sr>_denoised.wav`
- Sample-rate constants appear as literals per call site (22050 vs 44100) — check caller expectations before changing defaults
- Background sounds for augmentation live in `data/sound/background_sounds/mono_5s/`
