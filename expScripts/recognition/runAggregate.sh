#!/usr/bin/env bash

# The purpose of this code is to run code like this `sbatch aggregate.sh 0111171 neurosketch schaefer2018 15` for all 3 sub and all number of ROIs
# and then load the accuracy result of this code and compare them.

# subjects="0110171 0110172 0111171"
toml=$1
hemis="lh rh"
dataSource=realtime
recogExpFolder=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/

roiloc=wang
for num in {1..50};
do
    sbatch ${recogExpFolder}aggregate.sh $toml $dataSource $roiloc $num
    echo sbatch ${recogExpFolder}aggregate.sh $toml $dataSource $roiloc $num
done

roiloc=schaefer
for num in {1..300};do
    sbatch ${recogExpFolder}aggregate.sh $toml $dataSource $roiloc $num
    echo sbatch ${recogExpFolder}aggregate.sh $toml $dataSource $roiloc $num
done


