from pathlib import Path

import logging
logger = logging.getLogger(__name__)


def append_suffix(fn: str, suffix: str) -> str:
    path = Path(fn)
    return str(path.with_stem(f'{path.stem}_{suffix}'))


def change_ext(fn: str, ext: str) -> str:
    path = Path(fn)
    return str(path.with_suffix(ext))
