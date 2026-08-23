# dataprocess/

Core library (the only Poetry package). Import as `from dataprocess.<sub> import ...`.

## STRUCTURE

| Subpackage | Contents |
|------------|----------|
| `sound/` | audio ops + vendored nussl → see `sound/AGENTS.md` |
| `cwt/` | `cwt2.py` CWT transform, `scalogram.py` scalogram rendering (~347 ln) |
| `util/` | `file.py` path/file helpers, `data_process.py` dataset splitting, `presentation.py` YAML meta (`Meta`, `load_meta`, `parse_meta`) |
| `image/` | `split.py` image splitting with overlap |

## CONVENTIONS

- Functions accept file paths OR numpy arrays; side effects = write files next to input unless output dir given
- Loose script-style strays at package root: `dwt-recon.py`, `multi-dwt.py`, `gbif-id-util.py` — legacy scripts, not part of the import surface
- Meta YAML schema (sep datasets) is defined by `util/presentation.Meta`; change both together

## ANTI-PATTERNS

- Don't add heavy deps for things librosa/scipy/PyWavelets already do
