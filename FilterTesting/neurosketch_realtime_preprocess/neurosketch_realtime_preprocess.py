# preprocess code for neurosketch data for testing different filters
# There are six run for each subject recognition task
# I use 5 runs to train the model and 1 run to test the model and that testing accuracy 
# is used as the metric. comparing different filtering methods.

# How should the metrics be? Should it be model testing accuracy of training process 

# import and set up environment
import sys
from subprocess import call
import nibabel as nib
import pydicom as dicom
import numpy as np
import time
import os
import glob
import shutil

def recognition_dataAnalysis_brain(sub='0110171_neurosketch',run=1, templateFunctionalVolume=None): # normally sub should be sub001
	print('sub=',sub)
	print('run=',run)
	homeDir="/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/" 
	dataDir=f"{homeDir}subjects/{sub}/data/nifti/"

	# if the data have been analyzed, load the saved data.
	PreprocessedData=f'{dataDir}realtime_preprocessed/{sub}_recognition_run_{run}.nii.gz'
	if not os.path.isdir(f'{dataDir}realtime_preprocessed/'):
		os.mkdir(f'{dataDir}realtime_preprocessed/')

	if os.path.exists(PreprocessedData):
		return

	tmp_folder = f"/gpfs/milgram/scratch60/turk-browne/kp578/sandbox/{sub}_run{run}" # tmp_folder='/tmp/kp578/'
	if os.path.isdir(tmp_folder):
		shutil.rmtree(tmp_folder)

	if not os.path.isdir(tmp_folder):
		os.mkdir(tmp_folder) # the intermediate data are saved in scratch folder, only original data and final results are saved in project folder

	def printOrien(full_ref_BOLD):
		# input might be .nii.gz file
		ref_BOLD_obj = nib.load(full_ref_BOLD)
		ref_bold_ornt = nib.aff2axcodes(ref_BOLD_obj.affine)
		print('Ref BOLD orientation:') # day1functionalInAnatomicalSpace Ref BOLD orientation:('R', 'A', 'S') # What is the orientation of the realtime data? 
		print(ref_bold_ornt)
		return ref_bold_ornt

	functional=f'{dataDir}{sub}_recognition_run_{run}.nii.gz'

	# copy the functional data to tmp folder 
	call(f"cp {functional} {tmp_folder}/", shell=True)

	# split functional data to multiple volumes
	call(f"fslsplit {tmp_folder}/{functional.split('/')[-1]}  {tmp_folder}/",shell=True)

	if not os.path.exists(tmp_folder+'/original_functional/'):
		os.mkdir(tmp_folder+'/original_functional/') # the intermediate data are saved in scratch folder, only original data and final results are saved in project folder
	call(f"mv {tmp_folder}/{functional.split('/')[-1]} {tmp_folder+'/original_functional/'}",shell=True)

	functionalFiles=glob.glob(f'{tmp_folder}/*.nii.gz')
	functionalFiles.sort()
	print('functionalFiles=',functionalFiles)

	if run==1:
		# select the middle volume as the template functional volume
		templateFunctionalVolume=functionalFiles[int(len(functionalFiles)/2)]
		call(f"cp {templateFunctionalVolume} \
			{dataDir}realtime_preprocessed/templateFunctionalVolume.nii.gz",shell=True)
	else:
		templateFunctionalVolume=f"{dataDir}realtime_preprocessed/templateFunctionalVolume.nii.gz"

	print('templateFunctionalVolume=',templateFunctionalVolume) #/gpfs/milgram/scratch60/turk-browne/kp578/sandbox/0110171_neurosketch_run1/0005.nii.gz

	# align every other functional volume with templateFunctionalVolume
	outputFileNames=[]
	for curr_TR in functionalFiles:
		print('curr_TR=',curr_TR)
		TR_FunctionalTemplateSpace=f"{curr_TR[0:-7]}_FunctionalTemplateSpace.nii.gz"
		command = f"3dvolreg \
		-base {templateFunctionalVolume} \
		-prefix  {TR_FunctionalTemplateSpace} \
		-1Dfile {curr_TR[0:-7]}_motion.1D -1Dmatrix_save {curr_TR[0:-7]}_mat.1D {curr_TR}"
		print('running'+command)
		A=time.time()
		call(command,shell=True)
		B=time.time()
		print(f"{B-A}s passed")
		outputFileNames.append(TR_FunctionalTemplateSpace)

	# merge the aligned data to the PreprocessedData, finish preprocessing
	files=''
	for f in outputFileNames:
	    files=files+' '+f
	command=f"fslmerge -t {PreprocessedData} {files}"
	print('running',command)
	call(command, shell=True)

	if os.path.exists(PreprocessedData):
		print(f"{PreprocessedData} exists")
	else:
		print(f"{PreprocessedData} does not exist!")
		error
	return templateFunctionalVolume

sub=sys.argv[1] # sub='0110171_neurosketch'
for run in range(1,7):
	recognition_dataAnalysis_brain(sub=sub,run=run)

# generate transformation matrix from functional template space to anatomical space
templateFunctionalVolume=f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}/data/nifti/realtime_preprocessed/templateFunctionalVolume.nii.gz'
templateFunctionalVolume_bet=f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}/data/nifti/realtime_preprocessed/templateFunctionalVolume_bet.nii.gz'
call(f"bet {templateFunctionalVolume} {templateFunctionalVolume_bet}",shell=True)
AnatomicalFile=f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}/data/nifti/{sub}_anat_mprage_brain.nii.gz'
templateFunctionalVolume_inAnatSpace=templateFunctionalVolume_bet[:-7]+'_inAnatSpace.nii.gz'
command = f'flirt -in {templateFunctionalVolume_bet} -ref {AnatomicalFile} \
-out {templateFunctionalVolume_inAnatSpace} -omat /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}/data/nifti/realtime_preprocessed/func2anat.mat'
print('Running ' + command)
call(command, shell=True)


# ## - to run all the subjects
# # bash to submit jobs
# # run neurosketch_realtime_preprocess parallelly

# # neurosketch_realtime_preprocess_child.sh
# #!/bin/bash
# #SBATCH --partition=short,scavenge
# #SBATCH --job-name rt_sketch
# #SBATCH --time=3:00:00
# #SBATCH --output=logs/rt_sketch-%j.out
# #SBATCH --mem=50g
# #SBATCH --mail-type=FAIL
# module load miniconda
# module load AFNI/2018.08.28
# module load FSL
# source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.0-centos7_64/etc/fslconf/fsl.sh 
# module load miniconda 
# source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud
# sub=$1
# /usr/bin/time python -u /gpfs/milgram/project/turk-browne/projects/rtcloud_kp/FilterTesting/neurosketch_realtime_preprocess/neurosketch_realtime_preprocess.py $sub


# # neurosketch_realtime_preprocess_parent.py
# from glob import glob
# import os
# from subprocess import call
# subject_dir='/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/'
# subjects=glob(subject_dir+'*_neurosketch')
# subjects=[sub.split('/')[-1] for sub in subjects if sub.split('/')[-1][0]!='_']
# for sub in subjects:
# 	command=f'sbatch neurosketch_realtime_preprocess_child.sh {sub}'
# 	print(command)
# 	# call(command, shell=True)