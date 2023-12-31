#!/bin/bash

# Data directory
# DATASET_DIR='/home/kelelu/Third_Paper/Codebase/DCASE2019/dataset_root/'
DATASET_DIR='/Users/kehindeelelu/Library/CloudStorage/OneDrive-ClemsonUniversity/RESEARCH/09_September_2023/Code/SELD/data_prep/'


# Feature directory
# FEATURE_DIR='/home/kelelu/Third_Paper/Codebase/SELD/dataset_root/feature/'
FEATURE_DIR='/Users/kehindeelelu/Library/CloudStorage/OneDrive-ClemsonUniversity/RESEARCH/09_September_2023/Code/SELD/dataset_root/feature/'

# Workspace
# WORKSPACE='/home/kelelu/Third_Paper/Codebase/SELD/'
WORKSPACE='/Users/kehindeelelu/Library/CloudStorage/OneDrive-ClemsonUniversity/RESEARCH/09_September_2023/Code/SELD/'
cd $WORKSPACE

# feature types: 'logmelgcc' | 'logmel'
FEATURE_TYPE='logmelgcc'

# audio types: 'mic' | 'foa'
AUDIO_TYPE='foa'

# Extract Features
python utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --data_type='dev' --audio_type=$AUDIO_TYPE
