#!/bin/bash

# Check if two command line arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./copy_n_times.sh <filename> <num_copies>"
    exit 1
fi

# Extract the filename and number of copies from the command line arguments
filename=$1
num_copies=$2

# Extract the file extension and base filename
extension="${filename##*.}"
base_filename="${filename%.*}"

# Loop through and create the copies
for i in $(seq -f "%02g" 1 $num_copies)
do
    copy_filename="$base_filename$i.$extension"
    cp "$filename" "$copy_filename"
done

