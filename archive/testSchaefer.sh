
' The purpose of this code is to follow the instructions of Jeff on the Schaefer and the wang2014 ROI selection code on the neurosketch dataset.' 
'Later this code would be adapted to Yuexuan data.'


# where the ROI selection code system is 
# cd /gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/

# bash make-wang-rois.sh 0110171  # this gives you the wang atlas in the subejct functional space in the current working dir.

# cd /gpfs/milgram/project/turk-browne/projects/rtTest/
# bash batchRegions.sh  # this would run classRegion.sh, which would run classRegion.py

# where to find subject name
# cd /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects





# how to apply the process on all available neurosketch dataset?

# get subjects ID: python
    di="/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/"
    from glob import glob
    subs=glob(f"{di}[0,1]*_neurosketch")
    subs=[sub.split("/")[-1].split("_")[0] for sub in subs]
    subjects=""
    for sub in subs:
        subjects=subjects+sub+" "
# resulting subjects is :
# subjects = "1206161 0119173 1206162 1201161 0115174 1130161 1206163 0120171 0111171 1202161 1121161 0125172 0110172 0123173 0120172 0113171 0115172 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0119171 0117171 0119174 0112173 0112174 0125171 0112172"

# step1: make-wang-rois.sh  &  make-schaefer-rois.sh
cd wang2014
bash make-wang-rois_parent.sh
    # make-wang-rois_parent.sh
    # #!/usr/bin/env bash
    # subjects="1206161 0119173 1206162 1201161 0115174 1130161 1206163 0120171 0111171 1202161 1121161 0125172 0110172 0123173 0120172 0113171 0115172 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0119171 0117171 0119174 0112173 0112174 0125171 0112172"
    # for sub in $subjects;do
    #     sbatch make-wang-rois.sh $sub
    # done
cd schaefer2018
bash make-schaefer-rois_parent.sh
    # make-schaefer-rois.sh
    # #!/usr/bin/env bash
    # subjects="1206161 0119173 1206162 1201161 0115174 1130161 1206163 0120171 0111171 1202161 1121161 0125172 0110172 0123173 0120172 0113171 0115172 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0119171 0117171 0119174 0112173 0112174 0125171 0112172"
    # for sub in $subjects;do
    #     sbatch make-schaefer-rois.sh $sub
    # done


# step2: batchRegions.sh  runs a runwise cross-validated classifier across the runs of recognition data, then stores the average accuracy of the ROI it was assigned in an numpy array
cd rtTest
bash batchRegions.sh


# step3: sbatch aggregate.sh 0111171 neurosketch schaefer2018 15
bash runAggregate.sh
