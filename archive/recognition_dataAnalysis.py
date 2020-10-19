# test recognition_dataAnalysis.py

## - behavior data analysis
import pandas as pd
workingDir='/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtcloud_kp/'
subjectID='test'
run=1
datapath=f'recognition/data/recognition/{subjectID}_{run}_.csv'
behav_data=pd.read_csv(workingDir+datapath)

# the item(imcode) colume of the data represent each image in the following correspondence
imcodeDict={'A': 'bed',
'B': 'Chair',
'C': 'table',
'D': 'bench'}

# When the imcode code is "A", the correct response should be '1', "B" should be '2'
correctResponseDict={'A': 1,
'B': 2,
'C': 1,
'D': 2}

# extract the labels which is selected by the subject and coresponding TR and time
behav_data = behav_data[['TR', 'image_on', 'Resp',  'Item']] # the TR, the real time it was presented, 
behav_data=behav_data.dropna(subset=['Item'])

# check if the subject's response is correct. When Item is A,bed, response should be 1, or it is wrong
isCorrect=[]
for curr_trial in range(behav_data.shape[0]):
    isCorrect.append(correctResponseDict[behav_data['Item'].iloc[curr_trial]]==behav_data['Resp'].iloc[curr_trial])
behav_data['isCorrect']=isCorrect # merge the isCorrect clumne with the data dataframe
behav_data=behav_data[behav_data['isCorrect']] #discard the trials where the subject made wrong selection

labels=[]
# get the labels I need for the output of this function
for curr_trial in range(behav_data.shape[0]):
    labels.append(imcodeDict[behav_data['Item'].ilod[curr_trial]])


# pretend that I have the corresponding brain_data which is M TR x N voxels
brain_data
# brain data is first aligned by pushed back 2TR(4s)
Brain_TR=np.arange(brain_data.shape[0])
Brain_TR = Brain_TR+2

# select volumes of brain_data by counting which TR is left in behav_data
Brain_TR=Brain_TR[list(behav_data['TR'])]
brain_data=brain_data[Brain_TR]

# This M x N brain_data and M labels


return brain_data, labels



################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
# This part of code load anatomical data and functional data align them together.

from subprocess import call
import nibabel as nib
from nibabel.nicom import dicomreaders
import pydicom as dicom
import numpy as np
import time
import os
import glob
import shutil

exampleFolder='/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/'
dicomFolder='/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/20190925.saCPM_sub01_day1.saCPM_sub01_day1/'

class MyClass:
    def __init__(self):
        pass

def dicom2nii(filename):
	dicomObject = dicom.read_file(filename)
	niftiObject = dicomreaders.mosaic_to_nii(dicomObject)
	print(nib.aff2axcodes(niftiObject.affine))
	splitList=filename.split('/')
	fullNiftiFilename='/'+os.path.join(*splitList[0:-1] , splitList[-1].split('.')[0]+'.nii.gz')
	print('fullNiftiFilename=',fullNiftiFilename)
	niftiObject.to_filename(fullNiftiFilename)
	return fullNiftiFilename

realtimeAlignmentMethod = 3dvolreg
print(f"{realtimeAlignmentMethod}")

# using the data from neurosketch data, from subject 0110171_neurosketch
Neurosketch_folder='/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/data/'
AnatomicalFile='/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/data/nifti/0110171_neurosketch_anat_mprage_brain.nii.gz'
day1RecognitionRunFile='/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/0110171_neurosketch021_0063_001'
day2RecognitionRunFile='/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/0110171_neurosketch021_0021_001'
day2RealtimeFolder='/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch'

# realtimeDicomFileNamePattern="{subject_ID}_neurosketch{runID}_{curr_TR}_001".format(subject_ID=0110171,runID=001-057,curr_TR=0001-0216)
realtimeDicomFileNamePattern="{subject_ID}_neurosketch{runID}_{curr_TR}_001".format(subject_ID='0110171',runID='001',curr_TR='0001') # 0110171_neurosketch021_0063_001

tmp_folder='/tmp/kp578/'
if os.path.isdir(tmp_folder):
	shutil.rmtree(tmp_folder)
os.mkdir(tmp_folder)

# align day1RecognitionRunFile with subject anatomy
A=time.time()
day1RecognitionRunFile=dicom2nii(day1RecognitionRunFile)
day1RecognitionRunFile_aligned=day1RecognitionRunFile[0:-7]+'_aligned.nii.gz'
# command = f"mcflirt -in {nii} -reffile {nii_ref} -out {nii_aligned} -plots"
command = f'flirt -in {day1RecognitionRunFile} -ref {AnatomicalFile} -out {day1RecognitionRunFile_aligned} -omat {tmp_folder}day1RecognitionRunFile_2_AnatomicalFile.mat'
print('Running ' + command) # flirt -in /gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/0110171_neurosketch021_0063_001.nii.gz -ref /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/data/nifti/0110171_neurosketch_anat_mprage_brain.nii.gz -out /gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/0110171_neurosketch021_0063_001_aligned.nii.gz -omat /tmp/kp578/day1RecognitionRunFile_2_AnatomicalFile.mat
call(command, shell=True)
B = time.time()
print('time=',B-A)

# align day2RecognitionRunFile with day1RecognitionRunFile using flirt
A = time.time()
day2RecognitionRunFile=dicom2nii(day2RecognitionRunFile)
day2RecognitionRunFile_aligned=day2RecognitionRunFile[0:-7]+'_aligned.nii.gz'
# command = f"mcflirt -in {day2RecognitionRunFile} -reffile {day1RecognitionRunFile} -out {day2RecognitionRunFile_aligned} -plots"
command = f'flirt -in {day2RecognitionRunFile} -ref {day1RecognitionRunFile} -out {day2RecognitionRunFile_aligned} -omat {tmp_folder}day2RecognitionRunFile_2_day1RecognitionRunFile.mat' 
print('Running ' + command) # flirt -in /gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/0110171_neurosketch021_0021_001.nii.gz -ref /gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/0110171_neurosketch021_0063_001.nii.gz -out /gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/0110171_neurosketch021_0021_001_aligned.nii.gz -omat /tmp/kp578/day2RecognitionRunFile_2_day1RecognitionRunFile.mat
call(command, shell=True)
B = time.time()
print('time=',B-A) 

# align every new incoming realtime dicom file to the day2RecognitionRunFile file using mcflirt, 
# save transformation matrix, and transform them into day2RecognitionRunFile space.
files = glob.glob("/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/0110171_neurosketch050_00*_001")
files=files[:40]
data=[]
RealtimeAlignment_time=[]
saved_transformation_time=[]
for newDicomFile in files: # files = glob.glob("/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/*_001")
	print(newDicomFile)
	A = time.time()
	newDicomFile=dicom2nii(newDicomFile)
	newDicomFile_aligned2=tmp_folder+newDicomFile.split('/')[-1][0:-7]+'_aligned2.nii.gz' #aligned to day2 functional template run vloume

	if realtimeAlignmentMethod=='3dvolreg':
		command = f"3dvolreg -verbose -zpad 1 -base {day2RecognitionRunFile} -cubic -prefix {newDicomFile_aligned2} \
		-1Dfile {newDicomFile_aligned2[0:-7]}_motion.1D -1Dmatrix_save {newDicomFile_aligned2[0:-7]}_mat.1D {newDicomFile}"
	elif realtimeAlignmentMethod=='mcflirt_normcorr':
		command = f"mcflirt -in {newDicomFile} -reffile {day2RecognitionRunFile} -out {newDicomFile_aligned2} -plots -cost  normcorr"
	elif realtimeAlignmentMethod=='mcflirt_mutualinfo':
		command = f"mcflirt -in {newDicomFile} -reffile {day2RecognitionRunFile} -out {newDicomFile_aligned2} -plots -cost  mutualinfo"
	elif realtimeAlignmentMethod=='mcflirt_woods':
		command = f"mcflirt -in {newDicomFile} -reffile {day2RecognitionRunFile} -out {newDicomFile_aligned2} -plots -cost  woods"
	elif realtimeAlignmentMethod=='mcflirt_leastsquares':
		command = f"mcflirt -in {newDicomFile} -reffile {day2RecognitionRunFile} -out {newDicomFile_aligned2} -plots -cost  leastsquares"
	elif realtimeAlignmentMethod=='flirt':
		command = f'flirt -in {newDicomFile} -ref {day2RecognitionRunFile} -out {newDicomFile_aligned2}' 

	print('Running ' + command)
	call(command, shell=True)
	B = time.time()
	print('3dvolreg time=',B-A) 
	RealtimeAlignment_time.append(B-A)

	newDicomFile_aligned2 = nib.load(newDicomFile_aligned2)
	newDicomFile_aligned2 = newDicomFile_aligned2.get_data()
	data.append(newDicomFile_aligned2)

data=np.asarray(data)
print('data.shape=',data.shape)
np.save('/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/sample/data.npy',data)












