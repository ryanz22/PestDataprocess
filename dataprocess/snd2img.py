import sys
import os
import argparse
from timeit import default_timer as timer
from datetime import timedelta
import psutil

from cwt import batch_extract, scaleo_extract


CMAP = 'magma'
SR = 22050
TRAIN_DIR = 'data/sound/cornell-birdcall/train_audio'


def main(args):
    match args.command:
        case 'extract':
            out_dir = args.out_dir
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
            batch_extract(plist, out_dir, batch=3, thres=args.threshold, imgsize=args.imgsize)

            # 41s to finish
            # for f in plist:
            #     scaleo_extract(f, outdir=out_dir)

            # 30s to finish
            # ray.init(num_cpus=4)
            # # fn_id = ray.put()
            # ray.get([scaleo_extract.remote(ray.put(f), outdir=out_dir) for f in plist])
            end = timer()
            print(f'======== total time: {timedelta(seconds=end-start)}')
        case 'single_extract':
            f = args.file
            out_dir = args.out_dir
            if not os.path.exists(out_dir):
                print(f'create folder {out_dir}')
                os.mkdir(out_dir)
            else:
                print(f'folder {out_dir} exists')
            scaleo_extract(f, outdir=out_dir, thres=args.threshold, img_size=args.imgsize)
        case 'test':
            pass
        case _:
            print(f'unknown command [{args.command}]')

if __name__ == '__main__':
    print(f'python version is {sys.version_info}')
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        sys.exit('this program needs python 3.10 and above to run')

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    aparser = argparse.ArgumentParser(description='CWT Utility')
    sub = aparser.add_subparsers(dest='command', help='command to run', required=True)
    extract_arg = sub.add_parser('extract', help='extract signature of a batch')
    extract_arg.add_argument('-i', '--input_dir', type=str, help='input folder',
                          required=True)
    extract_arg.add_argument('-o', '--out_dir', type=str, help='output folder',
                          required=True)
    extract_arg.add_argument('-t', '--threshold', type=int, default=-30,
                        help='threshold in dB, default -30', required=True)
    extract_arg.add_argument('-s', '--imgsize', type=int, default=512,
                        help='image size, default 512', required=True)
    single_arg = sub.add_parser('single_extract', help='extract signature of a single file')
    single_arg.add_argument('-f', '--file', type=str, help='file path',
                          required=True)
    single_arg.add_argument('-o', '--out_dir', type=str, help='output folder',
                          required=True)
    single_arg.add_argument('-t', '--threshold', type=int, default=-30,
                        help='threshold in dB, default -30', required=True)
    single_arg.add_argument('-s', '--imgsize', type=int, default=512,
                        help='image size, default 512', required=True)
    test_arg = sub.add_parser('test', help='test functions')
    test_arg.add_argument('-d', '--dir', type=str, help='file directory',
                          required=True)
    # aparser.add_argument('-s', '--start', type=int, help='start page',
    #                      required=True)
    # aparser.add_argument('-o', '--output', type=str, help='output file name',
    #                      required=True)
    targs = aparser.parse_args()

    main(targs)
