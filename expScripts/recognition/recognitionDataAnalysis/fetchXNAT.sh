#!/usr/bin/env bash

#SBATCH --output=GetData-%j.out
#SBATCH -p short
#SBATCH -t 6:00:00
#SBATCH --mem 20GB
#SBATCH -n 1
module load XNATClientTools
sess_ID=$1
cd /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/

ArcGet -host https://xnat-milgram.hpc.yale.edu/ -u kailong -p 563214789Peng! -s $sess_ID
