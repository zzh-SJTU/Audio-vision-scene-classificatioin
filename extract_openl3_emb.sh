#!/bin/bash

if [ ! $# -eq 1 ]; then
    echo -e "Usage: $0 <raw_data_dir>"
    exit 1
fi

DATA=$1

cd data

# prepare data
python split_data.py --input_path "$DATA/evaluation_setup/fold1_train.csv" \
    --output_path "./evaluation_setup"

python extract_openl3_emb.py --input_path './evaluation_setup/train.csv' \
    --dataset_path $DATA \
    --output_path './feature' \
    --split 'train'

python extract_openl3_emb.py --input_path './evaluation_setup/val.csv' \
    --dataset_path $DATA \
    --output_path './feature' \
    --split 'val'

python extract_openl3_emb.py --input_path "$DATA/evaluation_setup/fold1_evaluate.csv" \
    --dataset_path $DATA \
    --output_path './feature' \
    --split 'test'

python compute_mean_std.py --input_path './feature'

cd ..
