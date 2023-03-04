#!/bin/bash
# ./apply_changes.sh -a ls -f "ls *.sh" -g "-l"
while getopts a:f:g: flag
do
    case "${flag}" in
        a) action=${OPTARG};;
        f) file=${OPTARG};;
        g) args=${OPTARG};;
    esac
done
echo "action: $action";
echo "file: $file";
echo "args: $args";

$file | while read fn
do 
    cmd="$action $args $fn"
	echo $cmd
	$cmd
done
