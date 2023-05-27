from dataclasses import dataclass
from pathlib import Path

import yaml

from returns.result import Result, safe, Failure, Success
from returns.pipeline import flow
from returns.pointfree import bind

import functional as pyfn

from IPython.display import display, Markdown
import librosa
import IPython
import matplotlib.pyplot as plt

from dataprocess.cwt.scalogram import plot_all


@dataclass
class Meta:
    title: str
    description: str
    src_cnt: int
    has_noise: bool
    mix_fn: Path
    s1_fn: Path
    s2_fn: Path
    s3_fn: Path | None
    est_s1_fn: Path
    est_s2_fn: Path
    est_s3_fn: Path | None
    noise_fn: Path | None


def load_meta(fp: Path) -> Result[Meta, Exception]:
    with open(fp, "r") as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        title = yaml_data["title"]
        description = yaml_data["description"]
        src_cnt = yaml_data["src_cnt"]
        noise = yaml_data["has_noise"]
        mix_fn = yaml_data["src_mappings"]["mix"]
        s1_fn = yaml_data["src_mappings"]["s1"]
        s2_fn = yaml_data["src_mappings"]["s2"]
        est_s1_fn = yaml_data["src_mappings"]["est_s1"]
        est_s2_fn = yaml_data["src_mappings"]["est_s2"]

        fn_list = [mix_fn, s1_fn, s2_fn, est_s1_fn, est_s2_fn]

        noise_fn = None
        if noise:
            noise_fn = yaml_data["src_mappings"]["noise"]
            fn_list.append(noise_fn)

        s3_fn, est_s3_fn = None, None
        if src_cnt == 3:
            s3_fn = yaml_data["src_mappings"]["s3"]
            est_s3_fn = yaml_data["src_mappings"]["est_s3"]
            fn_list.extend([s3_fn, est_s3_fn])

        parent = fp.parent
        ret = pyfn.seq(fn_list).filter(lambda f: not (parent / f).exists())
        if ret.non_empty():
            return Failure(Exception(ret.make_string("\n")))

        return Success(
            Meta(
                title,
                description,
                src_cnt,
                noise,
                parent / mix_fn,
                parent / s1_fn,
                parent / s2_fn,
                parent / s3_fn if s3_fn else None,
                parent / est_s1_fn,
                parent / est_s2_fn,
                parent / est_s3_fn if est_s3_fn else None,
                parent / noise_fn if noise_fn else None,
            )
        )


@safe
def src_plot(fp: Path) -> str:
    d, sr = librosa.load(fp, sr=None)
    display(IPython.display.Audio(data=d, rate=sr))

    plot_all(d, sr=sr, out_fn=None, threshold=-60, show_scale=True)
    plt.show()

    return "good"


def plot(meta: Meta) -> Result[str, Exception]:
    print(meta)
    display(Markdown(f"## {meta.title}"))
    display(Markdown(f"**{meta.description}**"))

    display(Markdown("### Mix sound"))
    src_plot(meta.mix_fn)

    display(Markdown("### Source one sound"))
    src_plot(meta.s1_fn)
    # d2, sr2 = librosa.load(meta.s1_fn, sr=None)
    # display(IPython.display.Audio(data=d2, rate=sr2))

    # plot_all(d2, sr=sr2, out_fn=None, threshold=-60, show_scale=True)
    # plt.show()

    display(Markdown("### Estimated one sound"))
    src_plot(meta.est_s1_fn)

    # d3, sr3 = librosa.load(meta.est_s1_fn, sr=None)
    # display(IPython.display.Audio(data=d3, rate=sr3))

    # plot_all(d3, sr=sr3, out_fn=None, threshold=-60, show_scale=True)
    # plt.show()

    return Success("plot good")


def process(in_dir: Path) -> Result[str, Exception]:
    META_FN = "meta.yaml"
    return flow(
        in_dir / META_FN,
        load_meta,
        bind(plot),
    )
