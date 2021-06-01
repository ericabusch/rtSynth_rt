#!/bin/bash
#SBATCH --partition=short   
#SBATCH --job-name=sMasks
#SBATCH --time=40:00
#SBATCH --output=./logs/wangMask-%j.out
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
THR=10 #threshhold
ROIS="roi1 roi2 roi3 roi4 roi5 roi6 roi7 roi8 roi9 roi10 roi11 roi12 roi13 roi14 roi15 roi16 roi17 roi18 roi19 roi20 roi21 roi22 roi23 roi24 roi25"

ROIpath=/gpfs/milgram/project/turk-browne/shared_resources/atlases/ProbAtlas_v4/subj_vol_all

stand2funcDirectly=1
if [ "$stand2funcDirectly" -eq 1 ]; then
    #register deskulled roi to individual subject t1
    WANG2FUNC=${recognition_dir}wang2func.mat
    TEMPLATE=${recognition_dir}templateFunctionalVolume.nii
    TEMPLATE_bet=${recognition_dir}templateFunctionalVolume_bet.nii
    bet ${TEMPLATE} ${TEMPLATE_bet}
    WANGINFUNC=${recognition_dir}wanginfunc.nii.gz
    # if [ -f "$WANG2FUNC" ]; then
    #     echo "xfm mat exists"
    # else 
    #     echo "xfm mat does not exist"
    flirt -ref $TEMPLATE_bet -in $STAND -omat $WANG2FUNC -out $WANGINFUNC
    # fi
else
    #register deskulled roi to individual subject t1
    WANG2ANAT=${recognition_dir}wang2anat.mat
    ANAT2FUNC=${recognition_dir}anat2func.mat
    WANG2FUNC=${recognition_dir}wang2func.mat
    ANAT=${recognition_dir}../anat/T1.nii
    ANAT_bet=${recognition_dir}../anat/T1_bet.nii
    bet ${ANAT} ${ANAT_bet}
    TEMPLATE=${recognition_dir}templateFunctionalVolume.nii
    TEMPLATE_bet=${recognition_dir}templateFunctionalVolume_bet.nii
    bet ${TEMPLATE} ${TEMPLATE_bet}
    WANGinANAT=${recognition_dir}WANGinANAT.nii.gz
    WANGinFUNC=${recognition_dir}WANGinFUNC.nii.gz
    ANATinFUNC=${recognition_dir}ANATinFUNC.nii.gz

    # wang to anat
    flirt -ref $ANAT_bet -in $STAND -omat $WANG2ANAT -out $WANGinANAT

    # anat to func
    flirt -ref $TEMPLATE_bet -in $ANAT_bet -omat $ANAT2FUNC -out $ANATinFUNC # flirt -ref $TEMPLATE_bet -in $ANAT_bet -omat $ANAT2FUNC -out $ANATinFUNC -dof 6

    # apply anat to func on wang_in_anat
    flirt -ref $TEMPLATE_bet -in $WANGinANAT -out $WANGinFUNC -applyxfm -init $ANAT2FUNC

    # combine wang2anat and anat2func to wang2func
    # convert_xfm -omat AtoC.mat -concat BtoC.mat AtoB.mat
    convert_xfm -omat $WANG2FUNC -concat $ANAT2FUNC $WANG2ANAT
    # fslview_deprecated $WANGinFUNC $TEMPLATE_bet
fi


# using saved transformation matrix, convert ROIs from wang2014 standard space to individual T1 space
for ROI in $ROIS; do
    for HEMI in lh rh; do
        INPUT=$ROIpath/perc_VTPM_vol_${ROI}_${HEMI}.nii.gz # wang2014 standard space
        OUTPUT=${mask_dir}/wang_${ROI}_${HEMI}.nii.gz #individual T1 space ROI outputs
        flirt -ref $TEMPLATE_bet -in $INPUT -out $OUTPUT -applyxfm -init $WANG2FUNC 
    done

    # merge the mask from two hemisphere for selected ROI
    left=${mask_dir}/wang_${ROI}_lh.nii.gz
    right=${mask_dir}/wang_${ROI}_rh.nii.gz
    # output=./${sub}_${ROI}_combined.nii.gz
    # fslmaths $left -add $right $output 
    fslmaths $left -thr $THR -bin $left #take threshhold and then bin the data
    fslmaths $right -thr $THR -bin $right
done 
