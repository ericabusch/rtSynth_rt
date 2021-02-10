#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/agg-%j.out
#SBATCH --job-name aggregate
#SBATCH --partition=short,day,scavenge
#SBATCH --time=1:00:00
#SBATCH --mem=10000
#SBATCH -n 5

# Set up the environment
# module load FSL/5.0.9
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
conda activate
# module load Python/Anaconda3
# module load FreeSurfer/6.0.0
# module load BXH_XCEDE_TOOLS
# module load brainiak
module load nilearn
# module load miniconda
source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud

toml=$1
dataSource=$2
roiloc=$3
Nregions=$4
recogExpFolder=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/
# Run the python scripts
echo python -u ${recogExpFolder}aggregate.py $toml $dataSource $roiloc $Nregions

python -u ${recogExpFolder}aggregate.py $toml $dataSource $roiloc $Nregions
