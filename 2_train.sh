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

FEATURE_TYPE='logmelgcc'
AUDIO_TYPE='foa'
SEED=10

# GPU number
GPU_ID=0

# Train SED
# TASK_TYPE: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
TASK_TYPE='sed_only'
for FOLD in {1..4}
    do
    echo $'\nFold: '$FOLD
    CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main.py train --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --seed=$SEED
done

# Train DOA
# TASK_TYPE: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
TASK_TYPE='doa_only'
for FOLD in {1..4}
    do
    echo $'\nFold: '$FOLD
    CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main.py train --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --seed=$SEED
done





