import os
import csv
import logging
import glob
import random
import shutil
import sys  # noqa F401
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from returns.result import Result, safe, Success, Failure

from dataprocess.util.file import list_subfolders
from dataprocess.util.data_process import split_list

logger = logging.getLogger(__name__)
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 44100


def prepare_audio_id_ds(
    data_folder: Path,
    save_folder: Path,
    split_ratio: tuple[float, float, float] = (0.7, 0.2, 0.1),
    dur: float = 2.0,
    mux: int = 1,
) -> Result[None, Exception]:
    """
    Prepares the csv files for the grasshopper datasets.
    Please follow the instructions in the README.md file for
    preparing Voxceleb2.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        txt file containing the verification split.
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    seg_dur : float
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).

    Example
    -------
    >>> from recipes.VoxCeleb.voxceleb1_prepare import prepare_voxceleb
    >>> data_folder = 'data/VoxCeleb1/'
    >>> save_folder = 'VoxData/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> prepare_voxceleb(data_folder, save_folder, splits, split_ratio)
    """

    # Setting output files
    save_csv_train = save_folder / TRAIN_CSV
    save_csv_dev = save_folder / DEV_CSV
    save_csv_test = save_folder / TEST_CSV

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev, wav_lst_test = get_gh_split_lists(
        data_folder, split_ratio
    )

    # Creating csv file for training data
    prepare_csv(data_folder, wav_lst_train, save_csv_train)

    prepare_csv(data_folder, wav_lst_dev, save_csv_dev)

    prepare_csv(data_folder, wav_lst_test, save_csv_test)

    return Success(None)


def get_gh_split_lists(
    data_folder: Path, split_ratio: tuple[float, float, float]
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Splits the audio file list into train and dev.
    This function automatically removes verification test files from the training and dev set (if any).
    """
    train_lst = []
    dev_lst = []
    test_lst = []

    for dir in list_subfolders(data_folder, "gh-*"):
        pattern = os.path.join(dir, "*.wav")
        wav_files = [Path(*(fn.split("/")[-2:])) for fn in glob.glob(pattern)]

        random.shuffle(wav_files)
        [train_l, dev_l, test_l] = split_list(wav_files, split_ratio)

        train_lst.extend(train_l)
        dev_lst.extend(dev_l)
        test_lst.extend(test_l)

    return train_lst, dev_lst, test_lst


def prepare_csv(data_folder: Path, wav_lst: list, csv_file):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file

    Returns
    -------
    None
    """
    import uuid

    msg = f"Creating csv lists in {csv_file}"
    logger.info(msg)

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    # For assigning unique ID to each chunk
    entry = []
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        spk_id = str(wav_file).split("/")[0]
        audio_id = f"{spk_id}-{uuid.uuid4()}"

        info = sf.info(str(data_folder / wav_file))
        audio_duration = info.duration
        start_sample = 0
        stop_sample = info.frames

        # Composition of the csv_line
        csv_line = [
            audio_id,
            str(audio_duration),
            f"$data_root/{wav_file}",
            start_sample,
            stop_sample,
            spk_id,
        ]
        entry.append(csv_line)

    csv_output = csv_output + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Final prints
    msg = f"{csv_file} successfully created!"
    logger.info(msg)
