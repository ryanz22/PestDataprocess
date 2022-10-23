import sys
import os
from timeit import default_timer as timer
from datetime import timedelta
import psutil
import click
import logging
from dataprocess.cwt.cwt2 import ( batch_extract, scaleo_extract, rd_file,
                                  cwt3, cwt2 )
from dataprocess.cwt.scalogram import plot_file
from dataprocess.util.data_process import replace_zeroes
from dataprocess.util.file import change_ext


CMAP = 'magma'
SR = 22050
TRAIN_DIR = 'data/sound/cornell-birdcall/train_audio'


@click.group()
def cli():
    pass


@cli.command()
@click.option('-i', '--in_dir', required=True,
            type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option('-o', '--out_dir', required=True, 
            type=click.Path(exists=False, dir_okay=True, file_okay=False))
@click.option('--threshold', type=int, default=-60)
@click.option('--imgsize', type=int, default=256)
def extract(in_dir: str, out_dir: str, threshold, imgsize):
    if not os.path.exists(out_dir):
        print(f'create folder {out_dir}')
        os.mkdir(out_dir)
    else:
        print(f'folder {out_dir} exists')

    flist = ['/btbwar/XC139608.mp3', '/btbwar/XC51863.mp3', '/btbwar/XC134502.mp3', '/btbwar/XC415596.mp3']
    plist = [ f'{TRAIN_DIR}{f}' for f in flist ]
    print(f'plist: {plist}')
    start = timer()
    core_cnt = psutil.cpu_count(logical=False)
    print(f'this computer has {core_cnt} physical cores')
    # 27s to finish
    batch_extract(plist, out_dir, batch=3, thres=threshold, imgsize=imgsize)

    # 41s to finish
    # for f in plist:
    #     scaleo_extract(f, outdir=out_dir)

    # 30s to finish
    # ray.init(num_cpus=4)
    # # fn_id = ray.put()
    # ray.get([scaleo_extract.remote(ray.put(f), outdir=out_dir) for f in plist])
    end = timer()
    print(f'======== total time: {timedelta(seconds=end-start)}')


@cli.command()
@click.option('-i', '--in_fn', required=True,
            type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.option('-o', '--out_dir', required=True, 
            type=click.Path(exists=False, dir_okay=True, file_okay=False))
@click.option('--threshold', type=int, default=-60)
@click.option('--imgsize', type=int, default=256)
def single_extract(in_fn: str, out_dir: str, threshold, imgsize):
    if not os.path.exists(out_dir):
        print(f'create folder {out_dir}')
        os.mkdir(out_dir)
    else:
        print(f'folder {out_dir} exists')
    scaleo_extract(in_fn, outdir=out_dir, thres=threshold, img_size=imgsize)


@cli.command()
@click.option('-i', '--in_fn', required=True,
            type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.option('-t', '--type', type=click.Choice(['waveshow', 'spectrogram', 'scalogram', 'all']), required=True)
@click.option('--threshold', type=int, default=-60)
def plot(in_fn: str, type: str, threshold):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    fig = plot_file(in_fn, type, threshold)
    out_fn = change_ext(in_fn, '.jpg')
    fig.savefig(out_fn)
    # plt.close()


@cli.command()
def test():
    pass


if __name__ == '__main__':
    print(f'python version is {sys.version_info}')
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        sys.exit('this program needs python 3.10 and above to run')

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    # print(f'sys.path:\n{sys.path}')

    l_fmt = '[%(name)s %(levelname)s] %(asctime)s - %(message)s'
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(l_fmt))
    logger = logging.getLogger('dataprocess')
    logger.addHandler(ch)
    logger.setLevel(logging.ERROR)

    cli()
