#!/usr/bin/env bash

#SBATCH --output=GetData-%j.out
#SBATCH -p short
#SBATCH -t 6:00:00
#SBATCH --mem 20GB
#SBATCH -n 1
module load XNATClientTools
sess_ID=$1
cd where-you-want-the-raw-data-to-be

ArcGet -host https://xnat-milgram.hpc.yale.edu/ -u kailong -p 563214789Peng! -s $sess_ID
