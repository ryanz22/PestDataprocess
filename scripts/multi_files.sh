#!/bin/bash
# TODO
# for some reason -f 'find data/sound/xeno-canto-acrididae -name "*.mp3"' doesn't work
# fail to handle when file name contains space
#
# Examples
#
# call 'app/snd2img.py plot -t scalogrram' to plot all wav files in 'data/sound/pest_2/'
# scripts/multi_files.sh -a app/snd2img.py -t 'plot' -g '-t scalogram -i' -f 'ls data/image/pest_2/**/**/*.wav'
#
# call 'app/snd_tool.py to-wav' to convert all mp3 found under 'data/sound/xeno-canto-acrididae' to wav 
# scripts/multi_files.sh -a app/snd_tool.py -t 'to-wav' -g '-f' -f 'ls data/sound/xeno-canto-acrididae/**/**/*.mp3'
#

while getopts t:a:f:g: flag
do
    case "${flag}" in
        t) task=${OPTARG};;
        a) app=${OPTARG};;
        f) file=${OPTARG};;
        g) args=${OPTARG};;
    esac
done
echo "app: $app";
echo "task: $task";
echo "file: $file";
echo "args: $args";

cur_dir=`pwd`
echo "current dir: $cur_dir"
export PYTHONPATH=. 
echo "$file | while read fn"
$file | while read fn # read line by line instead of word by word
#for fn in $($file)
do 
    tdir=$(dirname "$fn")
    tfn=$(basename "$fn")
    cmd="poetry run python $app $task $args $tdir/'$tfn'"
	echo -e "\n\n$cmd\n\n"
	$cmd
done
