#!/usr/bin/env bash

#SBATCH --output=./logs/GetData-%j.out
#SBATCH -p short
#SBATCH -t 6:00:00
#SBATCH --mem 20GB
#SBATCH -n 1
module load XNATClientTools
sess_ID=$1

pwd=$(pwd)
raw_dir=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/
cd ${raw_dir}
mv README.txt README_old.txt
ArcGet -host https://xnat-milgram.hpc.yale.edu/ -u kailong -p 563214789Peng! -s $sess_ID > ${raw_dir}${sess_ID}_run_name.txt

cd ${pwd}