#!/bin/bash

# # Define config file paths
# Default=/home/xucabis2/salt/logs/_final/MaskFormer_default_20240828-T113812/config.yaml
# GLS=/home/xucabis2/salt/logs/_final/MaskFormer_GLS_20240902-T084826/config.yaml
# STCH=/home/xucabis2/salt/logs/_final/MaskFormer_STCH_20240904-T023541/config.yaml
# UW=/home/xucabis2/salt/logs/_final/MaskFormer_UW_20240901-T003519/config.yaml
# DWA=/home/xucabis2/salt/logs/_final/MaskFormer_DWA_20240901-T005936/config.yaml
# RLW=/home/xucabis2/salt/logs/_final/MaskFormer_RLW_20240826-T234255/config.yaml
# FAMO=/home/xucabis2/salt/logs/_final/MaskFormer_FAMO_20240905-T210944/config.yaml

# # Create an array of config files
# config_files=($Default $GLS $STCH $UW $DWA $RLW $FAMO)

AlignedMTL=/home/xucabis2/salt/logs/_final/MaskFormer_AlignedMTL_20240903-T152849/config.yaml
CAGrad=/home/xucabis2/salt/logs/_final/MaskFormer_CAGrad_20240903-T152853/config.yaml
GradNorm=/home/xucabis2/salt/logs/_final/MaskFormer_GradNorm_20240903-T152853/config.yaml
NashMTL=/home/xucabis2/salt/logs/_final/MaskFormer_NashMTL_20240903-T152853/config.yaml
PCGrad=/home/xucabis2/salt/logs/_final/MaskFormer_PCGrad_20240903-T152853/config.yaml
GradVac=/home/xucabis2/salt/logs/_final/MaskFormer_GradVac_20240909-T132709/config.yaml

# Create an array of config files
config_files=($AlignedMTL $CAGrad $GradNorm $NashMTL $PCGrad $GradVac)

# Loop through each config file and run the command
for config in "${config_files[@]}"; do
    echo "Running salt test for config: $config"
    salt test -c "$config" --data.batch_size 10000 --data.test_file /home/xzcappon/phd/datasets/vertexing_120m/fixed_evaluation/output/pp_output_test_zprime.h5
done