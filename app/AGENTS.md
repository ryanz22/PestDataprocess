# app/

CLI layer. Each file is a standalone typer/click app run directly — never installed as console scripts.

## TOOLS

| File | Framework | Commands | Domain |
|------|-----------|----------|--------|
| `snd_tool.py` (~624 ln) | click | ~16 | denoise/mono/normalize/resample/filter/slice/stretch/mix/augment/to-wav/SNR |
| `dataset_tool.py` (~527 ln) | click | ~9 | split-folders, plot-all-wav, xeno-canto normalize/peaks, sep-dataset gen, audio-id dataset |
| `img_tool.py` (~494 ln) | click | ~10 | split/resize/flip/format/augment/crop image ops |
| `snd2img.py` (~320 ln) | click | ~5 | wav → cwt image extraction |
| `file_tool.py` | **typer** (only one) | 2 | cherry-pick wav files between folders |

## CONVENTIONS

- Command signature pattern: `-f FILE` single input or `-i DIR` recursive input; output beside input by default, `-o DIR` to redirect
- Commands are thin: parse args → call `dataprocess.*` functions. New processing logic belongs in the library, not here
- `ml/fingers_train.py` / `fingers_eval.py`: standalone torch experiments (finger counting) — unrelated to the grasshopper pipeline; don't model new ML code on them

## ANTI-PATTERNS

- Don't add a 6th framework choice: new CLIs use click (match majority), not typer
