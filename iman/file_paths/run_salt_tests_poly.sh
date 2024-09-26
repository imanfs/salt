#!/bin/bash

# Define file paths with descriptive names
Default_eps1=/home/xucabis2/salt/logs/_final/MaskFormer_PolyMaskCEeps1_20240905-T235531/config.yaml
Default_epsNeg1=/home/xucabis2/salt/logs/_final/MaskFormer_PolyMaskCE_20240905-T220809/config.yaml
DWA=/home/xucabis2/salt/logs/_final/MaskFormer_DWA_20240907-T012306/config.yaml
GLS=/home/xucabis2/salt/logs/_final/MaskFormer_GLS_20240907-T101337/config.yaml

# Create an associative array of descriptions and file paths
declare -A files=(
    ["Default (Poly-MaskCE, ε: 1)"]="$Default_eps1"
    ["Default (Poly-MaskCE, ε: -1)"]="$Default_epsNeg1"
    ["DWA (Poly-MaskCE, ε: -1)"]="$DWA"
    ["GLS (Poly-MaskCE, ε: -1)"]="$GLS"
)

# Loop through each file path and print the description and file path
for description in "${!files[@]}"; do
    file_path="${files[$description]}"
    echo "Processing: $description"
    echo "File path: $file_path"
    
    # Add your command here. For example, to run salt with the file:
    salt test -c "$file_path" --data.batch_size 10000 --data.test_file /home/xzcappon/phd/datasets/vertexing_120m/fixed_evaluation/output/pp_output_test_zprime.h5
done