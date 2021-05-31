#!/bin/bash
#SBATCH --output=ConvertDicom-%j.out
#SBATCH -p short
#SBATCH -t 6:00:00
#SBATCH --mem 20GB
#SBATCH -n 1
module load dcm2niix
pwd=$(pwd)
sess_ID=$1
export dcm_dir=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/${sess_ID}/SCANS/
export output_dir=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/${sess_ID}/nifti/
mkdir -p $output_dir
cd $dcm_dir

for k in *
do
    if [ -d "${k}" ]; then
        dcm2niix -o $output_dir -f %i_%t_%f ${dcm_dir}/$k
    fi
done

cd ${output_dir}
for k in *.nii ; do
    echo $k
    fslinfo $k  | grep dim4
done

for k in *.nii ; do
    echo $k
    fslinfo $k  | grep dim1
done

cd $pwd