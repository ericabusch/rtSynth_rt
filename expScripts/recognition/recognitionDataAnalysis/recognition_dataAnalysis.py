## - recognition data analysis for behavior data and brain data

## - How to run this code:
# milgram
# cd_rtcloud_kp
# cd expScripts/recognition/
# run
# activate_rt
# python recognition_dataAnalysis.py


# import and set up environment
from subprocess import call
import nibabel as nib
from nibabel.nicom import dicomreaders
import pydicom as dicom
import numpy as np
import time
import os
import glob
import shutil

def recognition_dataAnalysis_brain(sub='pilot_sub001',run=1,ses=1): # normally sub should be sub001
	'''
	steps:
		copy the functional data to tmp folder
		split functional data to multiple volumes
		select the middle volume as the template functional volume
		align every other functional volume with templateFunctionalVolume
		save all the functional data in day1 anatomical space, as M x N matrix. M TR, N voxels
	'''
	# This script is in the environment of '/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/expScripts/recognition/' 
	# and activate_rt
	# The purpose of this script is to analyze the brain data from recognition run

	homeDir="/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/" 
	dataDir=f"{homeDir}subjects/{sub}/ses{ses}_recognition/run{run}/nifti/"

	# # if the data have been analyzed, load the saved data.
	# if os.path.exists(dataDir+'recognitionData.npy'):
	# 	print(f'({dataDir}+recognitionData.npy exist')
	# 	recognitionData=np.load(dataDir+'recognitionData.npy')
	# 	return recognitionData

	tmp_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/sandbox/{sub}/"  # tmp_folder='/tmp/kp578/'
	if os.path.isdir(tmp_folder):
		shutil.rmtree(tmp_folder)
	os.mkdir(tmp_folder)

	def printOrien(full_ref_BOLD):
		# input might be .nii.gz file
		ref_BOLD_obj = nib.load(full_ref_BOLD)
		ref_bold_ornt = nib.aff2axcodes(ref_BOLD_obj.affine)
		print('Ref BOLD orientation:') # day1functionalInAnatomicalSpace Ref BOLD orientation:('R', 'A', 'S') # What is the orientation of the realtime data? 
		print(ref_bold_ornt)
		return ref_bold_ornt

	# # fetch data from XNAT
	if sub=='pilot_sub001':
		subjectID="rtSynth_pilot001"
		anatomical=dataDir+'rtSynth_pilot001_20201009165148_8.nii'
		functional=dataDir+'rtSynth_pilot001_20201009165148_13.nii'
	else:
		subjectID=f"rtSynth_{sub}"
		anatomical=dataDir+'rtSynth_{sub}_20201009165148_8.nii'
		functional=dataDir+'rtSynth_{sub}_20201009165148_13.nii'

	# call(f"sbatch fetchXNAT.sh {subjectID}",shell=True) # fetch data from XNAT
	# call(f"upzip {subjectID}.zip")
	# call(f"sbatch change2nifti.sh {subjectID}",shell=True) # convert dicom to nifti files

	# copy the functional data to tmp folder 
	command=f"cp {functional} {tmp_folder}"
	print(f"run {command}")
	call(command, shell=True)

	# split functional data to multiple volumes
	command=f"fslsplit {tmp_folder}{functional.split('/')[-1]}  {tmp_folder}"
	print(f"run {command}")
	call(command,shell=True) ## os.chdir(f"{tmp_folder}{subjectID}/")
	functionalFiles=glob.glob(f'{tmp_folder}*.nii.gz')
	functionalFiles.sort()
	print('functionalFiles=',functionalFiles)

	# select the middle volume as the template functional volume
	templateFunctionalVolume=functionalFiles[int(len(functionalFiles)/2)]
	command=f'cp {templateFunctionalVolume} {dataDir}/templateFunctionalVolume.nii.gz'
	print('running ',command)
	call(command,shell=True)

	# align every other functional volume with templateFunctionalVolume
	data=[]
	for curr_TR in functionalFiles:
		print('curr_TR=',curr_TR)
		TR_FunctionalTemplateSpace=f"{curr_TR[0:-7]}_FunctionalTemplateSpace.nii.gz"
		command = f"3dvolreg \
		-base {templateFunctionalVolume} \
		-prefix  {TR_FunctionalTemplateSpace} \
		{curr_TR}" # -1Dfile {curr_TR[0:-7]}_motion.1D -1Dmatrix_save {curr_TR[0:-7]}_mat.1D \
		print(command)
		A=time.time()
		call(command,shell=True)
		B=time.time()
		print(f"{B-A}s passed")
		TR_FunctionalTemplateSpace = nib.load(TR_FunctionalTemplateSpace)
		TR_FunctionalTemplateSpace = TR_FunctionalTemplateSpace.get_data()
		data.append(TR_FunctionalTemplateSpace)

	# note that this is not masked for now, so skull is included.
	# save all the functional data in day1 anatomical space, as M x N matrix. M TR, N voxels 
	recognitionData=np.asarray(data)
	recognitionData=recognitionData.reshape(recognitionData.shape[0],-1)
	print("shape of recognitionData=",recognitionData.shape)
	np.save(dataDir+'recognitionData.npy',recognitionData)


	return recognitionData


def recognition_dataAnalysis(sub='pilot_sub001',run=1,ses=1): # normally sub should be sub001
	'''
	purpose:
		process the brain and behavior data and save them for later use ()

	steps:
		extract the labels which is selected by the subject and coresponding TR and time
		check if the subject's response is correct. When Item is A,bed, response should be 1, or it is wrong
		brain data analysis using recognition_dataAnalysis_brain
		brain data is first aligned by pushed back 2TR(4s)
		select volumes of brain_data by counting which TR is left in behav_data
		create the META file
	'''
	# loading packages and general paths
	import pandas as pd
	import numpy as np
	import os

	if 'milgram' in os.getcwd():
		main_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/'
	else:
		main_dir='/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtcloud_kp/'

	datapath=main_dir+f'subjects/{sub}/ses{ses}_recognition/run{run}/{sub}_{run}.csv'
	behav_data=pd.read_csv(datapath)

	# the item(imcode) colume of the data represent each image in the following correspondence
	imcodeDict={
	'A': 'bed',
	'B': 'chair',
	'C': 'table',
	'D': 'bench'}

	# When the imcode code is "A", the correct response should be '1', "B" should be '2'
	correctResponseDict={
	'A': 1,
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
	behav_data['subj']=[sub for i in range(len(behav_data))]
	behav_data['run_num']=[int(run) for i in range(len(behav_data))]
	behav_data=behav_data[behav_data['isCorrect']] # discard the trials where the subject made wrong selection

	# labels=[]
	# # get the labels I need for the output of this function
	# for curr_trial in range(behav_data.shape[0]):
	#     labels.append(imcodeDict[behav_data['Item'].iloc[curr_trial]])

	## - brain data analysis
	brain_data = recognition_dataAnalysis_brain(sub=sub,run=run) # corresponding brain_data which is M TR x N voxels

	# for offline model, high-pass filtering and Kalman filtering should be implemented here.


	# brain data is first aligned by pushed back 2TR(4s)
	Brain_TR=np.arange(brain_data.shape[0])
	Brain_TR = Brain_TR+2

	# select volumes of brain_data by counting which TR is left in behav_data
	Brain_TR=Brain_TR[list(behav_data['TR'])]
	brain_data=brain_data[Brain_TR]

	# create the META file:

	# This M x N brain_data and M labels are brain_data and labels
	# The input of the model training function offlineModel.py is M x N brain data and M 
	# labels in the format of 
	brain_data_path=main_dir+f'subjects/{sub}/ses{ses}_recognition/run{run}/{sub}_{run}_preprocessed_brainData.npy'
	np.save(brain_data_path,brain_data)

	behav_data_path=main_dir+f'subjects/{sub}/ses{ses}_recognition/run{run}/{sub}_{run}_preprocessed_behavData.csv'
	behav_data.to_csv(behav_data_path)
	return brain_data, behav_data, brain_data_path, behav_data_path

sub='pilot_sub001'
run='01'
ses=1
for run in [1]:
	brain_data, behav_data, brain_data_path, behav_data_path = recognition_dataAnalysis(sub=sub,run=run,ses=ses)
print('behav_data=',behav_data)
print('brain_data.shape=',brain_data.shape)

# from offlineModel import offlineModel
# print('running model training')
# offlineModel(sub=sub,ses=ses,testRun=None, FEAT=brain_data, META=behav_data)



