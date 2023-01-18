#!/bin/bash
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

for fn in $($file)
do 
    $cmd = "$action $args $fn"
	echo $cmd
	$($cmd)
done
