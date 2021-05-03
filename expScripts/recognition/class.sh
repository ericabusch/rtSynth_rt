#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/class-%j.out
#SBATCH --job-name class
#SBATCH --partition=verylong,short,day,scavenge
#SBATCH --time=1:00:00 #20:00:00
#SBATCH --mem=10000
#SBATCH -n 5
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kp578

# Set up the environment
module load FSL
# module load Python/Anaconda3
# module load FreeSurfer/6.0.0
# module load BXH_XCEDE_TOOLS
# module load brainiak
# module load nilearn
source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud

echo python -u /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/class.py $1 $SLURM_ARRAY_TASK_ID
python -u /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/class.py $1 $SLURM_ARRAY_TASK_ID