from dataclasses import dataclass
from pathlib import Path

import yaml

from returns.result import Result, safe, Failure, Success
from returns.pipeline import flow
from returns.pointfree import bind

import functional as pyfn


@dataclass
class Meta:
    title: str
    description: str
    src_cnt: int
    has_noise: bool
    mix_fn: Path
    s1_fn: Path
    s1_desc: str | None
    s2_fn: Path
    s2_desc: str | None
    s3_fn: Path | None
    s3_desc: str | None
    est_s1_fn: Path
    est_s2_fn: Path
    est_s3_fn: Path | None
    noise_fn: Path | None


def load_meta(fp: Path) -> Result[Meta, Exception]:
    parent = fp.parent
    with open(fp, "r") as file:
        # yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        yaml_data = file.read()
        return parse_meta(yaml_data, parent)


def parse_meta(
    yaml_str: str, parent: Path, validate: bool = True
) -> Result[Meta, Exception]:
    yaml_data = yaml.safe_load(yaml_str)
    title = yaml_data["title"]
    description = yaml_data["description"]
    src_cnt = yaml_data["src_cnt"]
    noise = yaml_data["has_noise"]

    # for file existence valiation
    fn_list = []

    mix_fn, s1_fn, s2_fn, s1_desc, s2_desc = None, None, None, None, None
    est_s1_fn, est_s2_fn = None, None

    mappings = yaml_data["src_mappings"]
    for key in mappings:
        val = mappings.get(key, None)
        match key:
            case "mix":
                mix_fn = parent / val
                fn_list.append(mix_fn)
            case "s1":
                s1_fn = parent / val
                fn_list.append(s1_fn)
            case "s2":
                s2_fn = parent / val
                fn_list.append(s2_fn)
            case "est_s1":
                est_s1_fn = parent / val
                fn_list.append(est_s1_fn)
            case "est_s2":
                est_s2_fn = parent / val
                fn_list.append(est_s2_fn)
            case "s1_desc":
                s1_desc = val
            case "s2_desc":
                s2_desc = val
            case _:
                pass  # noise and s3 will be handled next

    noise_fn = None
    if noise:
        noise_fn = mappings.get("noise", None)
        if noise_fn is None:
            return Failure(Exception("Missing noise wav"))

        noise_fn = parent / noise_fn
        fn_list.append(noise_fn)

    s3_fn, est_s3_fn, s3_desc = None, None, None
    if src_cnt == 3:
        s3_fn = mappings.get("s3", None)
        if s3_fn is None:
            return Failure(Exception("Missing s3 wav"))

        s3_fn = parent / s3_fn

        s3_desc = mappings.get("s3_desc", None)

        est_s3_fn = mappings.get("est_s3", None)
        if est_s3_fn is None:
            return Failure(Exception("Missing est_s3 wav"))

        est_s3_fn = parent / est_s3_fn

        fn_list.extend([s3_fn, est_s3_fn])

    if validate:
        ret = pyfn.seq(fn_list).filter(lambda f: not f.exists())
        if ret.non_empty():
            return Failure(Exception(ret.make_string("\n")))

    return Success(
        Meta(
            title=title,
            description=description,
            src_cnt=src_cnt,
            has_noise=noise,
            mix_fn=mix_fn,
            s1_fn=s1_fn,
            s1_desc=s1_desc,
            s2_fn=s2_fn,
            s2_desc=s2_desc,
            s3_fn=s3_fn,
            s3_desc=s3_desc,
            est_s1_fn=est_s1_fn,
            est_s2_fn=est_s2_fn,
            est_s3_fn=est_s3_fn,
            noise_fn=noise_fn,
        )
    )


@safe
def src_plot(fp: Path) -> str:
    from IPython.display import Audio, display
    import IPython
    import librosa
    import matplotlib.pyplot as plt

    from dataprocess.cwt.scalogram import plot_all

    d, sr = librosa.load(fp, sr=None)
    display(IPython.display.Audio(data=d, rate=sr))

    plot_all(d, sr=sr, out_fn=None, threshold=-60, show_scale=True)
    plt.show()
    plt.close("all")

    return "good"


def plot(meta: Meta) -> Result[str, Exception]:
    # print(meta)
    from IPython.display import Markdown, display
    display(Markdown(f"## {meta.title}"))
    display(Markdown(f"**{meta.description}**"))

    display(Markdown("### Mix sound"))
    src_plot(meta.mix_fn)

    s1_desc = f" - {meta.s1_desc}" if meta.s1_desc else ""
    display(Markdown(f"### 1st source sound{s1_desc}"))
    src_plot(meta.s1_fn)

    display(Markdown(f"### 1st estimated sound{s1_desc}"))
    src_plot(meta.est_s1_fn)

    s2_desc = f" - {meta.s2_desc}" if meta.s2_desc else ""
    display(Markdown(f"### 2nd source sound{s2_desc}"))
    src_plot(meta.s2_fn)

    display(Markdown(f"### 2nd estimated sound{s2_desc}"))
    src_plot(meta.est_s2_fn)

    if meta.src_cnt == 3:
        s3_desc = f" - {meta.s3_desc}" if meta.s3_desc else ""
        display(Markdown(f"### 3rd source sound{s3_desc}"))
        if meta.s3_fn:
            src_plot(meta.s3_fn)

        display(Markdown(f"### 3rd estimated sound{s3_desc}"))
        if meta.est_s3_fn:
            src_plot(meta.est_s3_fn)

    if meta.has_noise:
        display(Markdown("### Noise sound"))
        src_plot(meta.noise_fn) if meta.noise_fn else None

    return Success("plot good")


def process(meta: Path) -> Result[str, Exception]:
    return flow(
        meta,
        load_meta,
        bind(plot),
    )
