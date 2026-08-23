import io

import pytest
from pathlib import Path

from dataprocess.util.file import (
    common_parent_path,
    extract_path_without_root,
    list_subfolders,
)

# https://medium.com/testcult/intro-to-test-framework-pytest-5b1ce4d011ae
# https://medium.com/beyn-technology/hands-on-start-testing-with-pytest-1ef39e59176a
# https://medium.com/ideas-at-igenius/fixtures-and-parameters-testing-code-with-pytest-d8603abb390a
# https://betterprogramming.pub/understand-5-scopes-of-pytest-fixtures-1b607b5c19ed


@pytest.mark.parametrize(
    "path1, path2, expected",
    [
        (
            Path("/root/directory1/subdirectory/file.txt"),
            Path("/root/directory2/subdirectory/file.txt"),
            Path("/root"),
        ),
        (
            Path("/root/directory1/subdirectory/file.txt"),
            Path("/root/directory1/subdirectory2/file.txt"),
            Path("/root/directory1"),
        ),
        (
            Path("/root/directory1/subdirectory/file.txt"),
            Path("/root/directory1/subdirectory2"),
            Path("/root/directory1"),
        ),
    ],
)
def test_common_parent_path(path1, path2, expected):
    assert common_parent_path(path1, path2) == expected


@pytest.mark.parametrize(
    "path1, expected",
    [
        (
            Path("/root/directory1/subdirectory/file.txt"),
            Path("/directory1/subdirectory/file.txt"),
        ),
        (
            Path("/root/directory1/subdirectory"),
            Path("/directory1/subdirectory"),
        ),
        (
            Path("root/directory1/subdirectory"),
            Path("directory1/subdirectory"),
        ),
    ],
)
def test_extract_path_without_root(path1, expected):
    assert extract_path_without_root(path1) == expected


@pytest.mark.parametrize(
    "root, pattern, expected",
    [
        (
            Path("ext-gh-class/grasshopper-aug-ds/"),
            "gh-*",
            [
                "ext-gh-class/grasshopper-aug-ds/gh-1",
                "ext-gh-class/grasshopper-aug-ds/gh-2",
                "ext-gh-class/grasshopper-aug-ds/gh-3",
                "ext-gh-class/grasshopper-aug-ds/gh-4",
                "ext-gh-class/grasshopper-aug-ds/gh-5",
                "ext-gh-class/grasshopper-aug-ds/gh-6",
                "ext-gh-class/grasshopper-aug-ds/gh-7",
                "ext-gh-class/grasshopper-aug-ds/gh-8",
                "ext-gh-class/grasshopper-aug-ds/gh-9",
                "ext-gh-class/grasshopper-aug-ds/gh-10",
                "ext-gh-class/grasshopper-aug-ds/gh-11",
                "ext-gh-class/grasshopper-aug-ds/gh-12",
                "ext-gh-class/grasshopper-aug-ds/gh-13",
                "ext-gh-class/grasshopper-aug-ds/gh-14",
                "ext-gh-class/grasshopper-aug-ds/gh-15",
                "ext-gh-class/grasshopper-aug-ds/gh-16",
            ],
        )
    ],
)
@pytest.mark.skipif(
    not Path("ext-gh-class/grasshopper-aug-ds").exists(),
    reason="requires ext-gh-class symlink target (external media) to exist",
)
def test_list_subfolders(root, pattern, expected):
    sub = list_subfolders(root, pattern)
    assert set(sub) == set(expected)
