#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/maskmaker-%j.out
#SBATCH --job-name searchlight
#SBATCH --partition=scavenge,short,day,verylong,long
#SBATCH --time=1:00:00
#SBATCH --mem=10000
##SBATCH -n 25

# Set up the environment
cd /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/
module load FSL/5.0.9
module load Python/Anaconda3
# module load FreeSurfer/6.0.0
# module load BXH_XCEDE_TOOLS
# module load brainiak
# module load nilearn
# module load miniconda

source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud
toml=$1
dataloc=$2
roiloc=$3
roinum=$4
roihemi=$5


# Run the python scripts
echo "running searchlight"
mkdir ./$roiloc/$subject/output

python -u ./classRegion.py $toml $dataloc $roiloc $roinum $roihemi
