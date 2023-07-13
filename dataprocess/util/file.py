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
    """_summary_
    https://www.geeksforgeeks.org/python-copy-directory-structure-without-files/

    Args:
        idir (str): _description_
        odir (str): _description_

    Returns:
        _type_: _description_
    """
    import shutil

    # defining the function to ignore the files
    # if present in any folder
    def ignore_files(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    # calling the shutil.copytree() method and
    # passing the src,dst,and ignore parameter
    shutil.copytree(idir, odir, ignore=ignore_files, dirs_exist_ok=True)


def common_parent_path(path1: Path, path2: Path) -> Path:
    return Path(os.path.commonpath([path1, path2]))


def extract_path_without_root(path1: Path) -> Path:
    # Extract the path parts
    parts = path1.parts
    print(parts)
    # Reconstruct the path without the root
    # new_path = Path("").joinpath(*parts[1:])
    if path1.is_absolute():
        new_path = "/" + "/".join(parts[2:])
    else:
        new_path = "/".join(parts[1:])

    return Path(new_path)
