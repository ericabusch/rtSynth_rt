#!/usr/bin/env bash
#SBATCH --output=logs/8runRecgnitionModelTraining-%j.out
##SBATCH -p day
##SBATCH -t 24:00:00
#SBATCH --partition=psych_day,psych_scavenge,psych_week,verylong,short,day,scavenge_all,week
#SBATCH --time=6:00:00 #20:00:00
#SBATCH --mem 4GB
#SBATCH -n 1
module load FSL
# module load miniconda
source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud


toml=$1 # sub002.ses5.toml
scan_asTemplate=$2 # 1
forceGreedy=$3 # 1 #skipPre=$3 # 1

if [ "$forceGreedy" == "1" ]; then
    echo "forceGreedy"
    python -u /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.py -c $toml --scan_asTemplate $scan_asTemplate --skipPre --forceGreedy
else
    echo "not forceGreedy"
    python -u /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.py -c $toml --scan_asTemplate $scan_asTemplate
fi
