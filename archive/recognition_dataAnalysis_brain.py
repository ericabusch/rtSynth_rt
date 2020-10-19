# imprt and set up environment
from subprocess import call
import nibabel as nib
from nibabel.nicom import dicomreaders
import pydicom as dicom  # type: ignore
import numpy as np
import time
import os
import glob
import shutil
homeDir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/'
dataDir=homeDir+'recognition/dataAnalysis/rtSynth_pilot001/nifti/'
tmp_folder = '/gpfs/milgram/scratch60/turk-browne/kp578/sandbox/' # tmp_folder='/tmp/kp578/'
if os.path.isdir(tmp_folder):
	shutil.rmtree(tmp_folder)
os.mkdir(tmp_folder)

# # fetch data from XNAT
subjectID="rtSynth_pilot001"
# call(f"sbatch fetchXNAT.sh {subjectID}",shell=True) # fetch data from XNAT
# call(f"upzip {subjectID}.zip")
# call(f"sbatch change2nifti.sh {subjectID}",shell=True) # convert dicom to nifti files

# convert functional run to anatomical space
anatomical=dataDir+'rtSynth_pilot001_20201009165148_8.nii'
functional=dataDir+'rtSynth_pilot001_20201009165148_13.nii'

# copy the functional data to tmp folder
os.mkdir(tmp_folder+subjectID)
call(f"cp {functional} {tmp_folder}{subjectID}/",shell=True)

# split functional data to multiple volumes
call(f"fslsplit {tmp_folder}{subjectID}/{functional.split('/')[-1]}  {tmp_folder}{subjectID}/",shell=True) ## os.chdir(f"{tmp_folder}{subjectID}/")
functionalFiles=glob.glob(f'{tmp_folder}{subjectID}/*.nii.gz')
functionalFiles.sort()

# select the middle volume as the template functional volume
templateFunctionalVolume=functionalFiles[int(len(functionalFiles)/2)]

# align the middle volume to the anatomical data and visually check the result # save the middle volume as day1functionalInAnatomicalSpace.nii.gz
day1functionalInAnatomicalSpace_bet=dataDir+"day1functionalInAnatomicalSpace_bet.nii.gz"
templateFunctionalVolume_bet=f"{templateFunctionalVolume[0:-7]}_bet.nii.gz"
call(f"bet {templateFunctionalVolume} {templateFunctionalVolume_bet}",shell=True)
anatomical_bet=f"{anatomical[0:-4]}_bet.nii.gz"
call(f"bet {anatomical} {anatomical_bet}",shell=True)
functional2anatomical=dataDir+'functional2anatomical'
call(f"flirt -in {templateFunctionalVolume_bet} -ref {anatomical_bet} \
	-omat {functional2anatomical}\
	-out {day1functionalInAnatomicalSpace_bet}",shell=True)

# transform templateFunctionalVolume with skull to anatomical space using saved transformation
day1functionalInAnatomicalSpace=dataDir+"day1functionalInAnatomicalSpace.nii.gz"
call(f"flirt -in {templateFunctionalVolume} -ref {anatomical} -out {day1functionalInAnatomicalSpace} \
-init {functional2anatomical} -applyxfm",shell=True) # the result of this looks fine, although not perfect.

# align every other functional volume with this day1functionalInAnatomicalSpace
data=[]
for curr_TR in functionalFiles:
	TRinAnatomical=f"{curr_TR[0:-7]}_AnatomicalSpace.nii.gz"
	command = f"3dvolreg -base {day1functionalInAnatomicalSpace}\
	-prefix  {TRinAnatomical} \
	-1Dfile {curr_TR[0:-7]}_motion.1D -1Dmatrix_save {curr_TR[0:-7]}_mat.1D {curr_TR}"
	print(command)
	A=time.time()
	call(command,shell=True)
	B=time.time()
	print(f"{B-A}s passed")
	TRinAnatomical = nib.load(TRinAnatomical)
	TRinAnatomical = TRinAnatomical.get_data()
	data.append(TRinAnatomical)

# save all the functional data in day1 anatomical space, as M x N matrix. M TR, N voxels
recognitionData=np.asarray(data)
recognitionData=recognitionData.reshape(recognitionData.shape[0],-1)
np.save(dataDir+'recognitionData',recognitionData)


