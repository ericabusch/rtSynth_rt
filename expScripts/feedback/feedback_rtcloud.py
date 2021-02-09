'''
steps:
	align new coming TR to the converted brain in the current day
	apply chosenMask
	load clf and get evidence
	pass evidence to morphing function and get the parameter
	pass the parameter to feedback receiver
	display code feedback.py receive the morphing parameter and morphs
'''

import os, glob, time, socket,shutil, sys,struct,math
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')
import nibabel as nib
from nibabel.nicom import dicomreaders
import numpy as np
from subprocess import call
from nilearn.image import new_img_like
from nilearn.input_data import NiftiLabelsMasker
import argparse
import rtCommon.utils as utils
from rtCommon.utils import loadConfigFile
import argparse
import scipy.io as sio
import pydicom as dicom  # type: ignore
import time
from glob import glob
import shutil
from nilearn.image import new_img_like
import joblib
import rtCommon.utils as utils
from rtCommon.utils import loadConfigFile
from rtCommon.fileClient import FileInterface
import rtCommon.projectUtils as projUtils
from rtCommon.imageHandling import readRetryDicomFromFileInterface, getDicomFileName, convertDicomImgToNifti

argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default='sub001.ses2.toml', type=str, help='experiment file (.json or .toml)')
args = argParser.parse_args()
from rtCommon.cfg_loading import mkdir,cfg_loading
cfg = cfg_loading(args.config)



## - function defintiion and environment setting up
def dicom2nii(filename):
	dicomObject = dicom.read_file(filename)
	niftiObject = dicomreaders.mosaic_to_nii(dicomObject)
	temp_data = niftiObject.get_data()
	output_image_correct = nib.orientations.apply_orientation(temp_data, ornt_transform)
	correct_object = new_img_like(templateFunctionalVolume, output_image_correct, copy_header=True)
	# print(nib.aff2axcodes(niftiObject.affine))
	splitList=filename.split('/')
	# fullNiftiFilename='/'+os.path.join(*splitList[0:-1] , splitList[-1].split('.')[0]+'.nii.gz')
	fullNiftiFilename=os.path.join(tmp_folder , splitList[-1].split('.')[0]+'.nii.gz')
	print('fullNiftiFilename=',fullNiftiFilename)
	correct_object.to_filename(fullNiftiFilename)
	return fullNiftiFilename

# def convertToNifti(tempNiftiDir,dicomObject,curr_dicom_name,ornt_transform,scratch_bold_ref):
#     nameToSaveNifti = curr_dicom_name.split('.')[0] + '.nii.gz'
#     fullNiftiFilename = os.path.join(tempNiftiDir, nameToSaveNifti)
#     niftiObject = dicomreaders.mosaic_to_nii(dicomObject)
#     print(nib.aff2axcodes(niftiObject.affine))
#     temp_data = niftiObject.get_data()
#     output_image_correct = nib.orientations.apply_orientation(temp_data, ornt_transform)
#     correct_object = new_img_like(scratch_bold_ref, output_image_correct, copy_header=True)
#     print(nib.aff2axcodes(correct_object.affine))
#     correct_object.to_filename(fullNiftiFilename)
#     print(fullNiftiFilename)
#     return fullNiftiFilename

# def getDicomFileName(cfg, scanNum, fileNum):
#     """
#     This function takes in different variables (which are both specific to the specific
#     scan and the general setup for the entire experiment) to produce the full filename
#     for the dicom file of interest.
#     Used externally.
#     """
#     if scanNum < 0:
#         raise ValidationError("ScanNumber not supplied or invalid {}".format(scanNum))

#     # the naming pattern is provided in the toml file
#     if cfg.dicomNamePattern is None:
#         raise InvocationError("Missing config settings dicomNamePattern")

#     if '{run' in cfg.dicomNamePattern:
#         fileName = cfg.dicomNamePattern.format(scan=scanNum, run=fileNum)
#     else:
#         scanNumStr = str(scanNum).zfill(2)
#         fileNumStr = str(fileNum).zfill(3)
#         fileName = cfg.dicomNamePattern.format(scanNumStr, fileNumStr)
#     fullFileName = os.path.join(cfg.dicomDir, fileName)

#     return fullFileName

tmp_folder=f'/gpfs/milgram/scratch60/turk-browne/kp578/{cfg.YYYYMMDD}.{cfg.LASTNAME}.{cfg.subjectName}/'

# if os.path.isdir(tmp_folder):
# 	shutil.rmtree(tmp_folder)
if not os.path.isdir(tmp_folder):
	os.mkdir(tmp_folder)

import random
# randomlist = []
# for i in range(0,50):
#     n = random.randint(1,19)
#     randomlist.append(n)
# print(randomlist)

scanNum = cfg.scanNum[0]
runNum = cfg.runNum[0]

# current TR dicom file name
SCANNUMBER=str(scanNum).zfill(6)
TRNUMBER=str(1).zfill(6)
dicomFileName = f"001_{SCANNUMBER}_{TRNUMBER}.dcm" # DICOM_file #SCANNUMBER might be run number; TRNUMBER might be which TR is this currently.

# this is the output of the recognition_dataAnalysis.py, meaning the day1 functional template volume in day1 anatomical space.
# day1functionalInAnatomicalSpace='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/recognition/dataAnalysis/rtSynth_pilot001/nifti/day1functionalInAnatomicalSpace.nii.gz'

realtime_ornt=nib.orientations.axcodes2ornt(('I', 'P', 'L'))
ref_ornt=nib.orientations.axcodes2ornt(('P', 'S', 'L'))
global ornt_transform
ornt_transform = nib.orientations.ornt_transform(realtime_ornt,ref_ornt)


for this_TR in np.arange(cfg.num_total_TRs):
	fileName = getDicomFileName(cfg, scanNum, this_TR) # get the filename expected for the new DICOM file, might be f"{subjectFolder}{dicomFileName}"
	# fileName="/gpfs/milgram/project/realtime/DICOM/20201009.rtSynth_pilot001.rtSynth_pilot001/001_000005_000149.dcm"
	print('fileName=',fileName)
	print("• use 'readRetryDicomFromFileInterface' to read dicom file for",
	    "TR %d, %s" %(this_TR, fileName)) # fileName is a specific file name of the interested file
	newDicomFile =  readRetryDicomFromFileInterface(fileInterface, fileName,timeout_file) # wait till you find the next dicom is available
	# newDicomFile=fileName
	newDicomFile=dicom2nii(newDicomFile) # convert dicom to nifti
	newDicomFile_aligned=tmp_folder+newDicomFile.split('/')[-1][0:-7]+'_aligned.nii.gz' #aligned to day1 functional template run volume in day1 anatomical space

	command = f"3dvolreg -base {templateFunctionalVolume} -prefix {newDicomFile_aligned} \
	-1Dfile {newDicomFile_aligned[0:-7]}_motion.1D -1Dmatrix_save {newDicomFile_aligned[0:-7]}_mat.1D {newDicomFile}"
	print('Running ' + command)
	A = time.time()
	call(command, shell=True)
	B = time.time()
	print('3dvolreg time=',B-A) 

	newDicomFile_aligned = nib.load(newDicomFile_aligned)
	newDicomFile_aligned = newDicomFile_aligned.get_data()
	newTR=newDicomFile_aligned.reshape(1,-1)
	print(newTR.shape)
	
	## - load the saved model and apply it on the new coming dicom file.
	# model_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/clf/'

	model_path=f"{cfg.subjects_dir}/{cfg.subjectName}/{cfg.session}_recognition/clf/" #/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/pilot_sub001/ses1_recognition/clf
	clf1 = joblib.load(model_path+'pilot_sub001_benchtable_tablebed.joblib') 
	clf2 = joblib.load(model_path+'pilot_sub001_benchtable_tablechair.joblib') 


	def gaussian(x, mu, sig):
		# mu and sig is determined before each neurofeedback session using 2 recognition runs.
		return round(20*(1 - np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))))

	# then do this for each TR
	s1 = clf1.score(newTR, ['table'])
	s2 = clf2.score(newTR, ['table'])
	NFparam = gaussian((s1 + s2)/2) # or an average or whatever
	print(NFparam)
	parameter = NFparam
	
	## - send the output of the model to web.
	projUtils.sendResultToWeb(projectComm, runNum, int(this_TR), parameter)

def doRuns(cfg, fileInterface, subjInterface):
    # variables we'll use throughout
    scanNum = cfg.scanNum[0]
    runNum = cfg.runNum[0]

	
def main(argv=None):
    """
    This is the main function that is called when you run 'sample.py'.

    Here, you will load the configuration settings specified in the toml configuration
    file, initiate the class fileInterface, and then call the function 'doRuns' to
    actually start doing the experiment.
    """

    # define the parameters that will be recognized later on to set up fileIterface
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                           help='experiment config file (.json or .toml)')
    argParser.add_argument('--runs', '-r', default='', type=str,
                           help='Comma separated list of run numbers')
    argParser.add_argument('--scans', '-s', default='', type=str,
                           help='Comma separated list of scan number')

    args = argParser.parse_args(argv)

    # Initialize the RPC connection to the projectInterface
    # This will give us a fileInterface for retrieving files and
    # a subjectInterface for giving feedback
    clientRPC = clientInterface.ClientRPC()
    fileInterface = clientRPC.fileInterface
    subjInterface = clientRPC.subjInterface

    # load the experiment configuration file
    cfg = loadConfigFile(args.config)

    # obtain paths for important directories (e.g. location of dicom files)
    if cfg.imgDir is None:
        cfg.imgDir = os.path.join(currPath, 'dicomDir')
    cfg.codeDir = currPath



    # now that we have the necessary variables, call the function 'doRuns' in order
    #   to actually start reading dicoms and doing your analyses of interest!
    #   INPUT:
    #       [1] cfg (configuration file with important variables)
    #       [2] fileInterface (this will allow a script from the cloud to access files
    #               from the stimulus computer that receives dicoms from the Siemens
    #               console computer)
    #       [3] subjInterface (to send/receive feedback to the subject in the scanner)
    doRuns(cfg, fileInterface, subjInterface)

    return 0


if __name__ == "__main__":
    """
    If 'sample.py' is invoked as a program, then actually go through all of the portions
    of this script. This statement is not satisfied if functions are called from another
    script using "from sample.py import FUNCTION"
    """
    main()
    sys.exit(0)
