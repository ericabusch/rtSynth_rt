# This script is run in day 2 recognition run, pick up a dicom in realtime folder
# in Milgram. And choose that as the day2 functional template volume. 
# register this day2 functional template volume with day1 functional template 
# volume and save that in day2 in day1 space.


# This script also process the two day 2 recognition run and get the metric floor and ceil (in practice calculate the sigma and mu of the metric transformation function)
# The relationship between floor, ceil  and mu, sigma is 

########################################################
########################################################

# floor=1
# ceil=-1

# mu = (floor+ceil)/2
# sig = (floor-ceil)/2.3548

# x=np.arange(-3,3,0.01)
# y=gaussian(x, mu, sig)

# plt.plot(x,y)

########################################################
########################################################

# The way to calculate floor and ceil is 
# Floor is C evidence for CD classifier (can also be D evidence for CD classifier, they are effectively the same)
# Ceil is A evidence in AC and AD classifier.

########################################################
########################################################

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import scipy.io as sio
from subprocess import call
from nibabel.nicom import dicomreaders
import pydicom as dicom  # type: ignore
import time
from glob import glob
import shutil
from nilearn.image import new_img_like
import joblib
# os.chdir('/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/')
sys.path.append('/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/')
# import project modules from rt-cloud
import rtCommon.utils as utils
from rtCommon.utils import loadConfigFile
from rtCommon.fileClient import FileInterface
import rtCommon.projectUtils as projUtils
from rtCommon.imageHandling import readRetryDicomFromFileInterface, getDicomFileName, convertDicomImgToNifti


def dicom2nii(templateVolume, filename,templateFunctionalVolume):
    dicomObject = dicom.read_file(templateVolume)
    niftiObject = dicomreaders.mosaic_to_nii(dicomObject)
    # print(nib.aff2axcodes(niftiObject.affine))
    temp_data = niftiObject.get_data()
    output_image_correct = nib.orientations.apply_orientation(temp_data, ornt_transform)
    correct_object = new_img_like(templateFunctionalVolume, output_image_correct, copy_header=True)
    print(nib.aff2axcodes(correct_object.affine))
    splitList=filename.split('/')
    # fullNiftiFilename="/".join(splitList[0:-1])+'/'+splitList[-1].split('.')[0]+'.nii.gz'
    fullNiftiFilename=os.path.join(tmp_folder, splitList[-1].split('.')[0]+'.nii.gz')
    print('fullNiftiFilename=',fullNiftiFilename)
    correct_object.to_filename(fullNiftiFilename)
    return fullNiftiFilename


# fetch the data folder
tomlFIle=f"/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/conf/tProject.toml"
argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default=tomlFIle, type=str, help='experiment file (.json or .toml)')
args = argParser.parse_args()
cfg = utils.loadConfigFile(args.config)

# YYYYMMDD= '20201009' #'20201009' '20201015'
# YYYYMMDD= '20201019' #'20201009' '20201015'
# LASTNAME='rtSynth_pilot001'
# PATIENTID='rtSynth_pilot001'
YYYYMMDD= cfg.YYYYMMDD #'20201009' '20201015'
LASTNAME=cfg.realtimeFolder_subjectName
PATIENTID=cfg.realtimeFolder_subjectName
Top_directory = '/gpfs/milgram/project/realtime/DICOM/'
sub='pilot_sub001'


realtime_ornt=nib.orientations.axcodes2ornt(('I', 'P', 'L'))
ref_ornt=nib.orientations.axcodes2ornt(('P', 'S', 'L'))
global ornt_transform
ornt_transform = nib.orientations.ornt_transform(realtime_ornt,ref_ornt)

tmp_folder=f'/gpfs/milgram/scratch60/turk-browne/kp578/{YYYYMMDD}.{LASTNAME}.{PATIENTID}/'
# if os.path.isdir(tmp_folder):
#   shutil.rmtree(tmp_folder)
if not os.path.isdir(tmp_folder):
    os.mkdir(tmp_folder)



subjectFolder=f"{Top_directory}{YYYYMMDD}.{LASTNAME}.{PATIENTID}/" #20190820.RTtest001.RTtest001: the folder for current subject # For each patient, a new folder will be generated:


dicomFiles=glob(f"{subjectFolder}*")
day2templateVolume_dicom=dicomFiles[int(len(dicomFiles)/2)]
day1templateFunctionalVolume='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/pilot_sub001/ses1_recognition/run1/nifti/templateFunctionalVolume.nii.gz'
day2templateVolume_fileName='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/pilot_sub001/ses2_recognition/templateFunctionalVolume.nii.gz'
day2templateVolume_nii=dicom2nii(day2templateVolume_dicom, day2templateVolume_fileName,day1templateFunctionalVolume) # convert dicom to nifti
# templateVolume=dicom2nii(templateVolume)

main_folder='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/'
day2functionalTemplate=f"{main_folder}subjects/{sub}/ses2_recognition/functionalTemplate.nii.gz"
call(f"cp {day2templateVolume_nii} {day2functionalTemplate}",shell=True)

day2functionalTemplate_inDay1=f"{main_folder}subjects/{sub}/ses2_feedback/day2functionalTemplate_inDay1.nii.gz"
command = f'flirt -in {day2functionalTemplate} \
-ref {day1templateFunctionalVolume} \
-out {day2functionalTemplate_inDay1}' 

print(command)
call(command,shell=True)


########################################################
########################################################

# floor=1
# ceil=-1

# mu = (floor+ceil)/2
# sig = (floor-ceil)/2.3548

# x=np.arange(-3,3,0.01)
# y=gaussian(x, mu, sig)

# plt.plot(x,y)

########################################################
########################################################

# The way to calculate floor and ceil is 
# Floor is C evidence for CD classifier (can also be D evidence for CD classifier, they are effectively the same)
# Ceil is A evidence in AC and AD classifier.


## - load the saved model and apply it on the new coming dicom file.
model_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/clf/'
clf1 = joblib.load(model_dir+'pilot_sub001_benchtable_tablebed.joblib') 
clf2 = joblib.load(model_dir+'pilot_sub001_benchtable_tablechair.joblib') 


def gaussian(x, mu, sig):
    # mu and sig is determined before each neurofeedback session using 2 recognition runs.
    return round(20*(1 - np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))))



A = "bed"
B = "chair"
C = "bench"
D = "table"

# then do this for each TR
s1 = clf1.score(newTR, ['table'])
s2 = clf2.score(newTR, ['table'])


