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


def copy_dir_only(idir: str, odir: str):
    import shutil

    # defining the function to ignore the files
    # if present in any folder
    def ignore_files(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    # calling the shutil.copytree() method and
    # passing the src,dst,and ignore parameter
    shutil.copytree(idir, odir, ignore=ignore_files)
