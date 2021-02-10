#!/bin/bash
#SBATCH --partition=short   
#SBATCH --job-name=sMasks
#SBATCH --time=40:00
#SBATCH --output=./logs/schaeferMask-%j.out
#SBATCH --mem=2g

'''
This script is adapted from /Users/kailong/Desktop/rtTest/schaefer2018/make-schaefer-rois.sh

Purpose: 
    get the customized masks for the current subject functional template masks

steps:
    convert the standard brain to the individual functional tempalte 

'''
module load AFNI/2018.08.28
module load FSL
source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.0-centos7_64/etc/fslconf/fsl.sh
module load miniconda
source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud


set -e #stop immediately encountering error
mkdir -p ./logs/
sub=$1 #sub001
recognition_dir=$2 #/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/sub001/ses1/recognition/
mask_dir=${recognition_dir}mask/
mkdir -p ${mask_dir} # save the output files in the current folder


STAND=/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_1mm_brain.nii.gz


ROIpath=/gpfs/milgram/scratch/turk-browne/tsy6/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI

#register deskulled roi to individual subject t1
WANG2FUNC=${recognition_dir}wang2func.mat
TEMPLATE=${recognition_dir}templateFunctionalVolume.nii
TEMPLATE_bet=${recognition_dir}templateFunctionalVolume_bet.nii
if [ -f "$TEMPLATE_bet" ]; then
    echo "TEMPLATE_bet mat exists"
else 
    echo "TEMPLATE_bet mat does not exist"
    bet ${TEMPLATE} ${TEMPLATE_bet}
fi
WANGINFUNC=${recognition_dir}wanginfunc.nii.gz
if [ -f "$WANG2FUNC" ]; then
    echo "xfm mat exists"
else 
    echo "xfm mat does not exist"
    flirt -ref $TEMPLATE_bet -in $STAND -omat $WANG2FUNC -out $WANGINFUNC
fi

atlas=Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.nii.gz

# using saved transformation matrix, convert schaefer ROIs from wang2014 standard space to individual T1 space
for ROI in {1..300}; do
  INPUT=$ROIpath/$atlas # schaefer2018 standard space
  OUTPUT=${mask_dir}/schaefer_${ROI}.nii.gz #individual T1 space ROI outputs
  fslmaths $INPUT -thr $ROI -uthr $ROI -bin $OUTPUT
  flirt -ref $TEMPLATE_bet -in $OUTPUT -out $OUTPUT -applyxfm -init $WANG2FUNC
  fslmaths $OUTPUT -thr 0.5 -bin $OUTPUT
done
  
