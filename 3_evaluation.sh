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


FEATURE_TYPE='logmelgcc'
AUDIO_TYPE='foa'
SEED=10

# TASK_TYPE: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
TASK_TYPE='doa_only'

# which model to use
ITERATION=9000

# GPU number
GPU_ID=0

############ Development Evaluation ############

# FOLD=4
# CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main.py inference --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --iteration=$ITERATION --seed=$SEED

# inference single fold
# for FOLD in {1..4}
#     do
#     echo $'\nFold: '$FOLD
#     CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main.py inference --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --iteration=$ITERATION --seed=$SEED
# done

# inference all folds
# python ${WORKSPACE}main.py inference_all --workspace=$WORKSPACE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --seed=$SEED


#===============================================

FOLD=3
CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main.py testaudio --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --iteration=$ITERATION --seed=$SEED
