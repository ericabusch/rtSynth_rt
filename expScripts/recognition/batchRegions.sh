#!/usr/bin/env bash

toml=$1
hemis="lh rh"

roiloc=wang
for hemi in $hemis;do
    for num in {1..25};do
        sbatch classRegion.sh $toml realtime $roiloc $num $hemi
        echo $toml realtime $roiloc $num $hemi
    done
done


roiloc=schaefer
for num in {1..300};do
    sbatch classRegion.sh $toml realtime $roiloc $num
    echo $toml realtime $roiloc $num
done

