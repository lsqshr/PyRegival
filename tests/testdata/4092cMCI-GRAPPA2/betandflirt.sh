#!/bin/bash
mkdir -p betted flirted

find ./ -name "*.nii" -print | while read f; do
    fname=$(basename $f)
    standard_space_roi  "$f" "./betted/$fname" -b
    bet "./betted/$fname" "./betted/$fname.betted.nii" -f 0.3
    rm "./betted/$fname.gz"
    flirt -ref /usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz -in "./betted/$fname.betted.nii" -out "./flirted/$fname.flirted.nii"
done


