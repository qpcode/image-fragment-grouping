#!/bin/bash
# Sample usage:
# ./train_sweep ../data/

# location of data folder has to be first argument
data_folder_path=$1

# first do sweep for linear models
for feature_dimension in 32 64 128 256 512
do
    python train.py --path-to-data-folder $data_folder_path --feature-dimension $feature_dimension --model-type linear
done

# now do sweep for conv models
for feature_dimension in 32 64 128 256
do
    python train.py --path-to-data-folder $data_folder_path --feature-dimension $feature_dimension --model-type conv
done
