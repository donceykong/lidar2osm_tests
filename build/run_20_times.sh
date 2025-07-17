#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <src_pcd_file> <tgt_pcd_file> <results_csv>"
    exit 1
fi

SRC_PCD=$1
TGT_PCD=$2
RESULTS_CSV=$3
RESOLUTION=1.0  # you can also parameterize this if desired

# Run the executable 20 times
for i in {1..20}
do
    echo "Iteration $i"
    ./Lidar2OSM "$SRC_PCD" "$TGT_PCD" "$RESULTS_CSV" "$RESOLUTION"
done

