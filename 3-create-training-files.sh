#!/bin/bash
set -x

echo "Creating training files.. "
python specter/data_utils/create_training_files.py \
  --data-dir data/training \
  --metadata data/training/metadata.json \
  --outdir data/preprocessed2/

echo "Done. See results is data/preprocessed2/"

