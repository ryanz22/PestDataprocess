# The dataset pipeline really cares 'id', 'mix_wav', 's1_wav', 's2_wav',
# 's3_wav' if mix3, 'noise_wav' if 'use_wham_noise'.
# pick 1K grasshopper 2 second sounds, two types of grasshopper, 1K bird
# 2 second sounds (any types), drone sound mix with environmental sound 1K.

import csv
import pathlib
import random
from enum import Enum

# from io import TextIOWrapper
from typing import TextIO
from functools import partial

from dataprocess.sound.preprocess import mix as lib_mix
from dataprocess.util.data_process import split_list

# GH1_DIR = "gh-18"
# GH1_DIR = "gh-mini"
# GH2_DIR = "gh-21"
# DIFF_GH_DIR = "gh-15"
# BIRD_DIR = "bird"
# DRONE_DIR = "drone"
# CRICKET_DIR = "cricket"

TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

DS_DIR_LIST = [TRAIN_DIR, VAL_DIR, TEST_DIR]

MIX2_CSV_COLUMNS = [
    "ID",
    "mix_wav",
    "s1_wav",
    "s2_wav",
    "noise_wav",
]

MIX3_CSV_COLUMNS = [
    "ID",
    "mix_wav",
    "s1_wav",
    "s2_wav",
    "s3_wav",
    "noise_wav",
]


def create_sep_dataset(
    savepath: pathlib.Path,
    main_src: pathlib.Path,
    mux: int,
    n_src: int,
    second_src: pathlib.Path,
    third_src: pathlib.Path,
    noise_src: pathlib.Path,
    train_ds: tuple[float, float, float] = None,
    fix_len: int = 0,
) -> Exception | None:
    """
    This functions creates the .csv file and sound mix for the src sep dataset

    mix strategy 1 - mix2 no noise, gh + bird: loop gh-18 (129 samples),
    random pick from bird (68 samples)

    CSV header
    ID, mix_wav, s1_wav, s2_wav

    mix strategy 2 - mix2 + noise, gh + bird + noise drone: loop gh-18 (129 samples),
    random pick from bird (68 samples) and drone (36 samples) as noise

    CSV header
    ID, mix_wav, s1_wav, s2_wav, noise_wav

    mix strategy 3 - mix3 no noise, gh + bird + drone: loop gh-18 (129 samples),
    random pick from bird (68 samples) and drone (36 samples)

    CSV header
    ID, mix_wav, s1_wav, s2_wav, s3_wav

    mix strategy 4 - mix3 no noise, gh + gh + drone: loop gh-18 (129 samples),
    random pick from gh-21 (82 samples) and drone (36 samples)

    CSV header
    ID, mix_wav, s1_wav, s2_wav, s3_wav

    mix strategy 5 - mix3 + noise, gh + gh + bird + noise drone: loop gh-18 (129 samples),
    random pick from gh-21 (82 samples), bird (68 samples) and drone (36 samples) as noise

    CSV header
    ID, mix_wav, s1_wav, s2_wav, s3_wav, noise_wav
    """
    csv_columns = MIX2_CSV_COLUMNS

    s1_path = main_src
    # s1_fl_paths = [f.name for f in s1_path.glob("*.wav")]
    s1_fl_paths = list(s1_path.glob("**/*.wav"))
    s1_fl_cnt = len(s1_fl_paths)
    if s1_fl_cnt == 0:
        return Exception(f"Can NOT find *.wav files in {s1_path}")

    print(f"total {len(s1_fl_paths)} files in {s1_path}")
    # print(f"S1 files:\n{s1_fl_paths}")

    s2_path = second_src

    # s2_fl_paths = [f.name for f in s2_path.glob("*.wav")]
    s2_fl_paths = list(s2_path.glob("**/*.wav"))
    s2_fl_cnt = len(s2_fl_paths)
    print(f"source 2 path: total {s2_fl_cnt} files in {s2_path}")
    if s2_fl_cnt == 0:
        return Exception(f"Can NOT find *.wav files in {s2_path}")
    # print(f"\n\nS2 files:\n{s2_fl_paths}")
    s3_fl_paths = []

    if n_src == 3:
        csv_columns = MIX3_CSV_COLUMNS
        s3_path = third_src

        # s3_fl_paths = [f.name for f in s3_path.glob("*.wav")]
        s3_fl_paths = list(s3_path.glob("**/*.wav"))
        s3_fl_cnt = len(s3_fl_paths)
        print(f"total {s3_fl_cnt} files in {s3_path}")
        if s3_fl_cnt == 0:
            return Exception(f"Can NOT find *.wav files in {s3_path}")

    if noise_src:
        # noise_fl_paths = [f.name for f in noise_path.glob("*.wav")]
        noise_fl_paths = list(noise_src.glob("**/*.wav"))
        noise_fl_cnt = len(noise_fl_paths)
        print(f"total {noise_fl_cnt} files in {noise_src}")
        # print(f"\n\nNoise files:\n{noise_fl_paths}")
        if noise_fl_cnt == 0:
            return Exception(f"Can NOT find *.wav files in {noise_src}")
    else:
        noise_fl_paths = []

    if train_ds is not None:
        random.shuffle(s1_fl_paths)
        ds_paths = split_list(s1_fl_paths, train_ds)

        ds_process_partial = partial(
            process,
            s2_fl_paths=s2_fl_paths,
            s3_fl_paths=s3_fl_paths,
            noise_fl_paths=noise_fl_paths,
            csv_columns=csv_columns,
            n_src=n_src,
            fix_len=fix_len,
            mux=mux,
        )

        add_noise = len(noise_fl_paths) > 0

        for dir_type, s1_paths in zip(DS_DIR_LIST, ds_paths):
            sub_dir = make_dir(savepath, dir_type, n_src, add_noise)
            with open(savepath / f"{dir_type}_mix_{n_src}.csv", "w") as sub_csv:
                ret = ds_process_partial(
                    s1_fl_paths=s1_paths,
                    csv_file=sub_csv,
                    savepath=sub_dir,
                )

            if isinstance(ret, Exception):
                return ret
    else:
        with open(savepath / f"mix_{n_src}.csv", "w") as csvfile:
            return process(
                csvfile,
                csv_columns=csv_columns,
                mux=mux,
                n_src=n_src,
                s1_fl_paths=s1_fl_paths,
                s2_fl_paths=s2_fl_paths,
                s3_fl_paths=s3_fl_paths,
                noise_fl_paths=noise_fl_paths,
                savepath=savepath,
                ds_mode="mono",
                fix_len=fix_len,
            )


def process(
    csvfile: TextIO,
    csv_columns: list[str],
    mux: int,
    n_src: int,
    s1_fl_paths: list[pathlib.Path],
    s2_fl_paths: list[pathlib.Path],
    s3_fl_paths: list[pathlib.Path],
    noise_fl_paths: list[pathlib.Path],
    savepath: pathlib.Path,
    ds_mode: str,
    fix_len: int,
) -> Exception | None:
    """
    sepformer requires all soundtrack must be exact same length
    """
    copy_file_fix_len = partial(copy_file, fix_len=fix_len)

    s1_fl_cnt = len(s1_fl_paths)
    if s1_fl_cnt == 0:
        return Exception("empty s1_fl_paths")

    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for n in range(0, mux):
        for i, path in enumerate(s1_fl_paths):
            id = n * s1_fl_cnt + i

            s1_wav = copy_file_fix_len(savepath, "s1", path)
            s2_wav = copy_file_fix_len(savepath, "s2", random_pick(s2_fl_paths))
            mix_wav = savepath / "mix" / f"mix_{id}.wav"

            row = {
                "ID": id,
                "mix_wav": f"$data_root/{ds_mode}/mix/" + str(mix_wav.name),
                "s1_wav": f"$data_root/{ds_mode}/s1/" + str(s1_wav.name),
                "s2_wav": f"$data_root/{ds_mode}/s2/" + str(s2_wav.name),
            }
            if n_src == 3:
                s3_wav = copy_file_fix_len(savepath, "s3", random_pick(s3_fl_paths))
                row["s3_wav"] = f"$data_root/{ds_mode}/s3/" + str(s3_wav.name)
            else:
                s3_wav = None

            if noise_fl_paths:
                noise_wav = copy_file_fix_len(
                    savepath, "noise", random_pick(noise_fl_paths)
                )
                row["noise_wav"] = f"$data_root/{ds_mode}/noise/" + str(noise_wav.name)
            else:
                noise_wav = None

            ret = mix(
                s1_wav,
                s2_wav,
                s3_wav,
                noise_wav,
                mix_wav,
                n_src,
            )
            if isinstance(ret, Exception):
                return ret

            writer.writerow(row)


def random_pick(fl: list[pathlib.Path]) -> None | pathlib.Path:
    cnt = len(fl)
    if cnt == 0:
        return None

    rand = random.randint(0, cnt - 1)

    return fl[rand]


def mix(
    s1_wav: pathlib.Path,
    s2_wav: pathlib.Path,
    s3_wav: pathlib.Path,
    noise_wav: pathlib.Path,
    mix_wav: pathlib.Path,
    n_src: int,
) -> Exception | None:
    """
    mix will generate mix soundtrack and copy source files
    """
    import soundfile as sf

    ret = lib_mix(str(s1_wav), str(s2_wav), mode="first")
    if isinstance(ret, Exception):
        return ret

    y_mix, sr = ret

    if n_src == 3:
        ret = lib_mix((y_mix, sr), str(s3_wav), mode="first")

        if isinstance(ret, Exception):
            return ret

        y_mix, sr = ret

    if noise_wav:
        ret = lib_mix((y_mix, sr), str(noise_wav), mode="first")

        if isinstance(ret, Exception):
            return ret

        y_mix, sr = ret

    sf.write(str(mix_wav), y_mix, sr)

    return None


def make_dir(
    path: pathlib.Path, dir_name: str, n_src: int, noise: bool
) -> pathlib.Path:
    new_path = path / dir_name
    new_path.mkdir()
    (new_path / "mix").mkdir()
    (new_path / "s1").mkdir()
    (new_path / "s2").mkdir()

    if n_src == 3:
        (new_path / "s3").mkdir()

    if noise:
        (new_path / "noise").mkdir()

    return new_path


def copy_file(
    dest_path: pathlib.Path, folder: str, src_file: pathlib.Path, fix_len: int
) -> pathlib.Path:
    import soundfile as sf
    import librosa
    import numpy as np

    dest_file = dest_path / folder / src_file.name  # Create the destination file path

    if not dest_file.exists():
        if fix_len > 0:  # make sure the dest soundtrack will be the exact len
            y, sr = librosa.load(str(src_file), sr=None, mono=True)
            y_len = len(y)
            if y_len > fix_len:
                y = y[:fix_len]
            elif y_len < fix_len:
                padding = fix_len - y_len
                y = np.pad(y, (0, padding), "constant")

            sf.write(str(dest_file), y, sr)
        else:
            dest_file.write_bytes(src_file.read_bytes())

    return dest_file


# def pick_src(src: str, datapath: pathlib.Path, gh_dir: str) -> pathlib.Path | Exception:
#     """
#     str: specify which types of soundtrack
#     gh_dir: point to the grasshopper folder
#     """
#     match src:
#         case "bird":
#             return datapath / BIRD_DIR
#         case "gh":
#             return datapath / gh_dir
#         case "cricket":
#             return datapath / CRICKET_DIR
#         case _:
#             return Exception(f"unknown source type: {src}")
