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
# which fsl:   /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/bin/fsl
# source       /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.0-centos7_64/etc/fslconf/fsl.sh
# module load miniconda
source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud


set -e #stop immediately encountering error
mkdir -p ./logs/
sub=$1 #sub=sub001
recognition_dir=$2 #recognition_dir=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/sub001/ses1/recognition/
mask_dir=${recognition_dir}mask/
mkdir -p ${mask_dir} # save the output files in the current folder


STAND=/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_1mm_brain.nii.gz


# ROIpath=/gpfs/milgram/scratch/turk-browne/tsy6/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI
ROIpath=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/MNI

stand2funcDirectly=0
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
    echo "running stand 2 anat 2 func"
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

    # anat to func, this was run in /Users/kailong/Desktop/rtEnv/rtSynth_rt/expScripts/recognition/makeGreyMatterMask.sh
    if [ -f "$ANAT2FUNC" ]; then
        echo "xfm mat exists"
    else 
        echo "xfm mat does not exist"
        flirt -ref $TEMPLATE_bet -in $ANAT_bet -omat $ANAT2FUNC -out $ANATinFUNC -dof 6 # flirt -ref $TEMPLATE_bet -in $ANAT_bet -omat $ANAT2FUNC -out $ANATinFUNC -dof 6

    # apply anat to func on wang_in_anat
    flirt -ref $TEMPLATE_bet -in $WANGinANAT -out $WANGinFUNC -applyxfm -init $ANAT2FUNC

    # combine wang2anat and anat2func to wang2func
    # convert_xfm -omat AtoC.mat -concat BtoC.mat AtoB.mat
    convert_xfm -omat $WANG2FUNC -concat $ANAT2FUNC $WANG2ANAT
    # fslview_deprecated $WANGinFUNC $TEMPLATE_bet
fi

atlas=Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.nii.gz

# using saved transformation matrix, convert schaefer ROIs from wang2014 standard space to individual T1 space
for ROI in {1..300}; do
    INPUT=$ROIpath/$atlas # schaefer2018 standard space
    OUTPUT=${mask_dir}/schaefer_${ROI}.nii.gz #individual T1 space ROI outputs
    fslmaths ${INPUT} -thr ${ROI} -uthr ${ROI} -bin ${OUTPUT}
    flirt -ref ${TEMPLATE_bet} -in ${OUTPUT} -out ${OUTPUT} -applyxfm -init ${WANG2FUNC}  
    echo fslmaths ${OUTPUT} -thr 0.5 -bin ${OUTPUT}
    fslmaths ${OUTPUT} -thr 0.5 -bin ${OUTPUT}
done

# GMINFUNC=${recognition_dir}../anat/gm_func.nii.gz
GMINFUNC=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/${sub}/ses1/anat/gm_func.nii.gz
for ROI in {1..300}; do  
    OUTPUT=${mask_dir}/schaefer_${ROI}.nii.gz
    GMmasked_OUTPUT=${mask_dir}/GMschaefer_${ROI}.nii.gz
    fslmaths ${OUTPUT} -add ${GMINFUNC} ${GMmasked_OUTPUT}
    echo fslmaths ${GMmasked_OUTPUT} -thr 1.5 -bin ${GMmasked_OUTPUT}
    fslmaths ${GMmasked_OUTPUT} -thr 1.5 -bin ${GMmasked_OUTPUT}
done

echo make-schaefer-rois.sh done


# 验证GM mask是成功的：
# for ROI in {1..300}; do  
#     temp=${mask_dir}/tmp.nii.gz
#     OUTPUT=${mask_dir}/schaefer_${ROI}.nii.gz
#     fslmaths ${OUTPUT} -add ${temp} ${temp}
# done

# for ROI in {1..300}; do  
#     GMtemp=${mask_dir}/GM_tmp.nii.gz
#     GMmasked_OUTPUT=${mask_dir}/GMschaefer_${ROI}.nii.gz
#     fslmaths ${GMmasked_OUTPUT} -add ${GMtemp} ${GMtemp}
# done

# fslview_deprecated $GMtemp $temp