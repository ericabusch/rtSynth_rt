# copy T1
cp subjects/sub002/ses1/anat/T1.nii ./archive/test_fsl/
# fslinfo T1.nii # 176 256 256

# copy gm
cp /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/sub002/ses1/GreyMatterMask/gm_shft_aligned_smooth+orig ./archive/test_fsl/
cd ./archive/test_fsl/
3dresample -input gm_shft_aligned_smooth+orig -prefix gm_anat.nii.gz
# fslinfo gm_anat.nii.gz # 256 256 256

3dresample -master T1.nii -prefix gm_anat.nii -input gm_shft_aligned_smooth+orig
# fslinfo gm_anat.nii # 176 256 256
# copy func
cp /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/sub001/ses1/recognition/templateFunctionalVolume.nii ./
# copy standard brain
cp /gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_1mm_brain.nii.gz ./stand.nii.gz
fslview_deprecated T1.nii gm_anat.nii
# perfect result
    
    

# project stand.nii.gz dierrectly to func space
stand=stand.nii.gz
WANG2FUNC=wang2func.mat
func=templateFunctionalVolume.nii
func_bet=templateFunctionalVolume_bet.nii
bet ${func} ${func_bet}
WANGINFUNC=wanginfunc.nii.gz
flirt -ref $func_bet -in $stand -omat $WANG2FUNC -out $WANGINFUNC

# result
fslview_deprecated $WANGINFUNC $func_bet 
# stand is fatter than func
        
        
project stand.nii.gz to T1 space and then to func
#register deskulled roi to individual subject t1
stand=stand.nii.gz
WANG2ANAT=wang2anat.mat
ANAT2FUNC=anat2func.mat
WANG2FUNC=wang2func.mat
ANAT=T1.nii
ANAT_bet=T1_bet.nii
bet ${ANAT} ${ANAT_bet}
FUNC=templateFunctionalVolume.nii
FUNC_bet=templateFunctionalVolume_bet.nii
bet ${FUNC} ${FUNC_bet}
WANGinANAT=WANGinANAT.nii.gz
WANGinFUNC=WANGinFUNC.nii.gz
ANATinFUNC=ANATinFUNC.nii.gz
# wang to anat
flirt -ref $ANAT_bet -in $stand -omat $WANG2ANAT -out $WANGinANAT
# anat to func
# flirt -ref $FUNC_bet -in $ANAT_bet -omat $ANAT2FUNC -out $ANATinFUNC -dof 6
flirt -ref $FUNC_bet -in $ANAT_bet -omat $ANAT2FUNC -out $ANATinFUNC
# apply anat to func on wang_in_anat
flirt -ref $FUNC_bet -in $WANGinANAT -out $WANGinFUNC -applyxfm -init $ANAT2FUNC
# combine wang2anat and anat2func to wang2func
# convert_xfm -omat AtoC.mat -concat BtoC.mat AtoB.mat
convert_xfm -omat $WANG2FUNC -concat $ANAT2FUNC $WANG2ANAT
fslview_deprecated $WANGinFUNC $FUNC_bet