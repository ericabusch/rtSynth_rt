#!/bin/bash
#SBATCH --output=ConvertDicom-%j.out
#SBATCH -p short
#SBATCH -t 6:00:00
#SBATCH --mem 20GB
#SBATCH -n 1
module load dcm2niix

sess_ID=$1
export dcm_dir=/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/recognition/dataAnalysis/${sess_ID}/SCANS/
export output_dir=/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/recognition/dataAnalysis/${sess_ID}/nifti/
mkdir $output_dir
cd $dcm_dir

for k in *
do
    if [ -d "${k}" ]; then
        dcm2niix -o $output_dir -f %i_%t_%f $dcm_dir/$k
    fi
done
