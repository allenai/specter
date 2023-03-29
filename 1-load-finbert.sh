#!/bin/bash 

echo "Starting to load finnish bert"
OUTPUT_DIR="hf_finbert"
mkdir $OUTPUT_DIR
python scripts/load_finbert.py $OUTPUT_DIR 
echo "Done".