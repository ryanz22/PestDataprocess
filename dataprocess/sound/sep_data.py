# The dataset pipeline really cares 'id', 'mix_wav', 's1_wav', 's2_wav',
# 's3_wav' if mix3, 'noise_wav' if 'use_wham_noise'.
# pick 1K grasshopper 2 second sounds, two types of grasshopper, 1K bird
# 2 second sounds (any types), drone sound mix with environmental sound 1K.

import csv
import pathlib
import random

from dataprocess.sound.preprocess import mix as lib_mix

GH1_DIR = "gh-18"
GH2_DIR = "gh-21"
BIRD_DIR = "bird"
DRONE_DIR = "drone"

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


def create_sep2mix_csv(
    datapath: pathlib.Path,
    savepath: pathlib.Path,
    mux: int,
    n_src: int,
    bird_or_gh: str = "bird",
    addnoise: bool = False,
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

    s1_path = datapath / GH1_DIR
    s1_fl_paths = list(s1_path.glob("*.wav"))
    s1_fl_cnt = len(s1_fl_paths)
    if s1_fl_cnt == 0:
        return Exception(f"Can NOT find *.wav files in {s1_path}")

    print(f"total {len(s1_fl_paths)} files in {s1_path}")
    # print(f"S1 files:\n{s1_fl_paths}")

    if n_src == 2:
        csv_columns = MIX2_CSV_COLUMNS
        if bird_or_gh == "bird":
            s2_path = datapath / BIRD_DIR
        else:
            s2_path = datapath / GH2_DIR

        s2_fl_paths = list(s2_path.glob("*.wav"))
        s2_fl_cnt = len(s2_fl_paths)
        print(f"total {s2_fl_cnt} files in {s2_path}")
        # print(f"\n\nS2 files:\n{s2_fl_paths}")
    else:
        csv_columns = MIX3_CSV_COLUMNS
        s2_path = datapath / BIRD_DIR
        s3_path = datapath / GH2_DIR

        s2_fl_paths = list(s2_path.glob("*.wav"))
        s2_fl_cnt = len(s2_fl_paths)
        print(f"total {s2_fl_cnt} files in {s2_path}")
        # print(f"\n\nS2 files:\n{s2_fl_paths}")

        s3_fl_paths = list(s3_path.glob("*.wav"))
        s3_fl_cnt = len(s3_fl_paths)
        print(f"total {s3_fl_cnt} files in {s2_path}")

    if addnoise:
        noise_path = datapath / DRONE_DIR
        noise_fl_paths = list(noise_path.glob("*.wav"))
        noise_fl_cnt = len(noise_fl_paths)
        print(f"total {noise_fl_cnt} files in {noise_path}")
        # print(f"\n\nNoise files:\n{noise_fl_paths}")
    else:
        noise_fl_paths = []

    with open(savepath / f"mix_{n_src}.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for n in range(0, mux):
            for i, path in enumerate(s1_fl_paths):
                id = n * s1_fl_cnt + i

                if n_src == 2:
                    row = {
                        "ID": id,
                        "mix_wav": savepath / f"mix_{id}.wav",
                        "s1_wav": path,
                        "s2_wav": random_pick(s2_fl_paths),
                        "noise_wav": random_pick(noise_fl_paths),
                    }
                else:
                    row = {
                        "ID": id,
                        "mix_wav": savepath / f"mix_{id}.wav",
                        "s1_wav": path,
                        "s2_wav": random_pick(s2_fl_paths),
                        "s3_wav": random_pick(s3_fl_paths),
                        "noise_wav": random_pick(noise_fl_paths),
                    }

                writer.writerow(row)

                ret = mix(row, n_src, addnoise)
                if isinstance(ret, Exception):
                    return ret

    return


def random_pick(fl: list[pathlib.Path]) -> pathlib.Path:
    cnt = len(fl)
    if cnt == 0:
        return pathlib.Path("")

    rand = random.randint(0, cnt - 1)

    return fl[rand]


def mix(row: dict, n_src: int, noise: bool) -> Exception | None:
    import soundfile as sf

    fn_1, fn_2 = str(row["s1_wav"]), str(row["s2_wav"])
    ret = lib_mix(fn_1, fn_2)
    if isinstance(ret, Exception):
        return ret

    y_mix, sr = ret

    if n_src == 3:
        ret = lib_mix((y_mix, sr), row["s3_wav"])

        if isinstance(ret, Exception):
            return ret

        y_mix, sr = ret

    if noise:
        ret = lib_mix((y_mix, sr), row["noise_wav"])

        if isinstance(ret, Exception):
            return ret

        y_mix, sr = ret

    sf.write(str(row["mix_wav"]), y_mix, sr)

    return None
