#!/bin/bash
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

$file | while read fn # read line by line instead of word by word
#for fn in $($file)
do 
    cmd="PYTHONPATH=. poetry run python $app $task $args $fn"
	echo $cmd
	$cmd
done
