#!/usr/bin/env bash

toml=$1
hemis="lh rh"

roiloc=wang
for hemi in $hemis;do
    for num in {1..25};do
        sbatch classRegion.sh $toml neurosketch $roiloc $num $hemi
        echo $sub neurosketch $roiloc $num $hemi
    done
done


roiloc=schaefer
for sub in $subjects;do
    for num in {1..300};do
    sbatch classRegion.sh $toml neurosketch $roiloc $num
    echo $sub neurosketch $roiloc $num
    done
done
