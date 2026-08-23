# PROJECT KNOWLEDGE BASE

**Generated:** 2026-08-22
**Commit:** c2647b4
**Branch:** main

## OVERVIEW

Audio-processing toolkit for pest detection via sound (grasshoppers/insects): preprocess audio, extract CWT/scalogram features, build ML datasets. Poetry + Python >=3.11,<3.13.

## STRUCTURE

```
PestDataprocess/
├── app/            # CLI tools (click, one typer); entry points — run directly, not installed scripts
├── dataprocess/    # Core library (`packages = [dataprocess]` in pyproject)
│   ├── sound/      # Audio preprocessing/augmentation/separation + VENDORED nussl subset
│   ├── cwt/        # Continuous wavelet transform + scalogram generation
│   ├── util/       # File/data helpers, YAML meta parsing
│   └── image/      # Image splitting with overlap
├── tests/          # pytest; run from repo root only
├── data/           # Datasets (partially git-tracked; see data/sound/README.md)
├── notebooks/      # Research notebooks (37 ipynb) — prototypes, not product code
├── scripts/        # Shell batch-runners (multi_files.sh runs a CLI over globbed files)
└── doc/            # Assets (logos, latex), not docs of this codebase
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/change an audio operation | `dataprocess/sound/preprocess.py` | mono/denoise/resample primitives |
| Add a CLI command | `app/snd_tool.py` or `app/dataset_tool.py` | `@cli.command()` pattern |
| Wavelet/scalogram feature extraction | `dataprocess/cwt/cwt2.py`, `scalogram.py` | plot type `-t scalogram` |
| Dataset split train/val/test | `dataprocess/util/data_process.py` → `dataset_tool.py split-folders` | |
| Source-separation datasets (mix/s1/s2) | `dataprocess/sound/sep_data.py` | meta YAML parsed by `util/presentation.Meta` |
| Dataset folder meanings | `data/sound/README.md` | authoritative per-folder descriptions |

## CONVENTIONS

- **`PYTHONPATH=.` is mandatory** for every python invocation (repo root on path). `.env` contains it.
- All commands go through poetry: `PYTHONPATH=. poetry run python app/<tool>.py <command>`
- CLIs process file-or-folder: `-f` single file, `-i` input dir, output written next to input unless `-o` given
- Type hints used loosely; black/pylint/mypy configured but minimal (`[mypy] mypy_path=.` is the entire mypy config)

## ANTI-PATTERNS (THIS PROJECT)

- **Do NOT edit `dataprocess/sound/nussl/`** — vendored subset of upstream nussl (pip dependency deliberately commented out at pyproject.toml:38). Treat read-only.
- Do NOT assume click everywhere: 4 CLIs are click, `app/file_tool.py` is typer.
- Do NOT import `nussl` from PyPI — the local vendored copy differs.

## UNIQUE STYLES

- Species folders named `gh-1` … `gh-26`; dataset lifecycle suffixes:
  `xxx-raw-ds` (cherry-picked peaks) → `xxx-aug-ds` (augmented) → `xxx-train-ds` (split) → `xxx-plot-ds` (CWT plots)
- Aug species folders must be created manually before augmenting; balance sample counts via `-c N`
- Symlinks `ext-data`, `ext-gh-class` point outside the repo to `/media/zhangjw/ml-data/...` — may dangle on other machines
- Repo carries large binaries at root (signal3.wav, soundscape.wav, IMG_1374.bmp) used as ad-hoc test inputs

## COMMANDS

```bash
# Run a CLI
PYTHONPATH=. poetry run python app/snd_tool.py --help

# Typical pipeline (see README.md for full xeno-canto workflow)
PYTHONPATH=. poetry run python app/dataset_tool.py xeno-canto-normalize -i <dir> -o <out> --sr 44100 --tsr 44100
PYTHONPATH=. poetry run python app/snd_tool.py augment -i <raw-ds>/gh-N/ -o <aug-ds>/gh-N -b data/sound/background_sounds/mono_5s/ -c 6
PYTHONPATH=. poetry run python app/dataset_tool.py split-folders -i <aug-ds> -o <train-ds>

# Tests (pytest style, no pytest config section exists)
PYTHONPATH=. poetry run pytest tests/test_snd_sep.py

# Quality
poetry run black .
poetry run pylint app/ dataprocess/
poetry run mypy app/ dataprocess/
```

## NOTES

- System deps required before `poetry install`: libsndfile, sox, cairo/gobject, ffmpeg (see README.md; Mac needs extra symlinks)
- basedpyright LSP server is configured but not installed — IDE diagnostics unavailable until `pip install basedpyright`
- Python constraint `>=3.11,<3.13` while local caches show 3.10–3.12 usage; pin to poetry's resolution
