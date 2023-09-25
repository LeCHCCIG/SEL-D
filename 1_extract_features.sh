#!/bin/bash

# Data directory
# DATASET_DIR='/home/kelelu/Third_Paper/Codebase/DCASE2019/dataset_root/'
# DATASET_DIR='/home/kelelu/Third_Paper/third/seld-dcase2019/dataset/'


# Feature directory
# FEATURE_DIR='/home/kelelu/Third_Paper/Codebase/SELD/dataset_root/feature/'
FEATURE_DIR='/Users/kehindeelelu/Documents/Research/SELD/dataset_root/feature/'

# Workspace
# WORKSPACE='/home/kelelu/Third_Paper/Codebase/SELD/'
WORKSPACE='/Users/kehindeelelu/Documents/Research/SELD/'
cd $WORKSPACE

# feature types: 'logmelgcc' | 'logmel'
FEATURE_TYPE='logmelgcc'

# audio types: 'mic' | 'foa'
AUDIO_TYPE='foa'

# Extract Features
python utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --data_type='dev' --audio_type=$AUDIO_TYPE
