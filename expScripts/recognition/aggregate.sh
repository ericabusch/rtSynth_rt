#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/agg-%j.out
#SBATCH --job-name aggregate
#SBATCH --partition=verylong
#SBATCH --time=20:00:00
#SBATCH --mem=10000
#SBATCH -n 5

# Set up the environment
module load FSL/5.0.9
module load Python/Anaconda3
module load FreeSurfer/6.0.0
module load BXH_XCEDE_TOOLS
module load brainiak
module load nilearn

toml=$1
dataSource=$2
roiloc=$3
Nregions=$4

# Run the python scripts
echo "running searchlight"

python -u ./aggregate.py $toml $dataSource $roiloc $Nregions
