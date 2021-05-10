#!/usr/bin/env bash
# Input python command to be submitted as a job
#SBATCH --output=logs/surf2func-%j.out
#SBATCH --job-name surf2func
#SBATCH --partition=short,scavenge,day
#SBATCH --time=2:00:00
#SBATCH --mem=10000
module load AFNI
module load FreeSurfer/6.0.0
module load FSL
. ${FSLDIR}/etc/fslconf/fsl.sh
'''
目的：将已经通过肉眼检查的functional space的grey matter mask与functional space的Schaefer mask进行overlap，然后获得去掉白质的mask

步骤：
    把anat投射到functional space
    然后用这个转移矩阵apply到surface上面，从而获得functional space的surface。
    最后用这个functional space的surface去mask所有的已经在functional space的SchaeferROI，并且保存在Schaefer的文件夹里面，命名为GM_${ROI}.nii.gz
'''

# 让当前的 makeGreyMatterMask.sh 在这个folder GreyMatterMask里面运行
subject=$1

# 查看一下在anat的皮层是否正确
GreyMatterMask_dir=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/${subject}/ses1/GreyMatterMask/
anat_reorien=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/sub001/ses1/anat/T1.nii
3dresample -master ${anat_reorien} -input ${GreyMatterMask_dir}gm_shft_aligned_smooth+orig -prefix ${GreyMatterMask_dir}gm_anat.nii.gz

# 制造从${anat_reorien}到functional space的转移矩阵
functional=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${subject}/neurosketch_recognition_run_1_bet.nii.gz
SURF2FUNC=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${subject}/surf2func.mat
SURFINFUNC=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${subject}/SURFinFUNC.nii.gz
flirt -ref $functional -in ${anat_reorien} -omat $SURF2FUNC -out $SURFINFUNC 

GMINFUNC=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${subject}/GMinFUNC.nii.gz
flirt -ref $functional -in ${GreyMatterMask_dir}gm_anat.nii.gz -out $GMINFUNC -applyxfm -init $SURF2FUNC

# 使用$GMINFUNC将schaefer的ROI进行mask，并且保存
for ROI in {1..300}; do
    echo processing ${ROI}
    currROI=/gpfs/milgram/project/turk-browne/projects/rtTest/schaefer2018/${subject}/${ROI}.nii.gz
    currROI_GM=/gpfs/milgram/project/turk-browne/projects/rtTest/schaefer2018/${subject}/GM_${ROI}.nii.gz
    fslmaths ${currROI} -add $GMINFUNC ${currROI_GM} #这里会出现一个提醒orientation不对的warning，不需要管他，我查看过是对的。
    fslmaths ${currROI_GM} -thr 1.5 -bin ${currROI_GM}
done


echo done



