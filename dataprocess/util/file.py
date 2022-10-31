from pathlib import Path
import os

import logging

logger = logging.getLogger(__name__)


def append_suffix(fn: str, suffix: str) -> str:
    path = Path(fn)
    return str(path.with_stem(f"{path.stem}_{suffix}"))


def change_ext(fn: str, ext: str) -> str:
    path = Path(fn)
    return str(path.with_suffix(ext))


def check_create_folder(dn: str) -> Path:
    if not os.path.exists(dn):
        logger.debug("create folder %s", dn)
        os.mkdir(dn)
    else:
        logger.debug("folder %s exists", dn)

    return Path(dn)
