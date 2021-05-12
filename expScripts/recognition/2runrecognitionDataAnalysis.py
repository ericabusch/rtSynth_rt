'''
purpose 
    to use 2 recognition runs in the current day and the saved model trained earlier 
        to get the functionalTemplateTR to align the feedback dicom to
        to align the selected functionalTemplateTR 
        register this day2 functional template volume with day1 functional template 
    to generate the metric Gaussian parameter and curve
'''

import os
import sys
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')
import argparse
import numpy as np
import nibabel as nib
import scipy.io as sio
from subprocess import call
from nibabel.nicom import dicomreaders
import pydicom as dicom  # type: ignore
import time
from scipy.stats import zscore
from glob import glob
import shutil
from nilearn.image import new_img_like
import joblib
import rtCommon.utils as utils
from rtCommon.utils import loadConfigFile
# from rtCommon.fileClient import FileInterface
import rtCommon.projectUtils as projUtils
# from rtCommon.imageHandling import readRetryDicomFromFileInterface, getDicomFileName, convertDicomImgToNifti

argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default='sub001.ses2.toml', type=str, help='experiment file (.json or .toml)')
argParser.add_argument('--skipPre', '-s', default=0, type=int, help='skip preprocess or not')
argParser.add_argument('--scan_asTemplate', '-t', default=1, type=int, help="which scan's middle dicom as Template?")

args = argParser.parse_args()
from rtCommon.cfg_loading import mkdir,cfg_loading
cfg = cfg_loading(args.config)

sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/')
from recognition_dataAnalysisFunctions import recognition_preprocess,minimalClass,behaviorDataLoading,recognition_preprocess_2run # morphingTarget,classifierEvidence


'''
convert all dicom files into nii files in the temp dir. 
find the middle volume of the run1 as the template volume
align every other functional volume with templateFunctionalVolume (3dvolreg)
load behavior data and align with brain data
'''
if not args.skipPre:
    scan_asTemplate=1 # which run in the realtime folder to select the middle volume as the template volume
    recognition_preprocess_2run(cfg,args.scan_asTemplate)

print(f"fslview_deprecated {cfg.templateFunctionalVolume_converted}")

'''
purpose:
    get the morphing target function
steps:
    load train clf
    load brain data and behavior data
    get the morphing target function
        evidence_floor is C evidence for CD classifier(can also be D evidence for CD classifier)
        evidence_ceil  is A evidence in AC and AD classifier
'''

# floor, ceil = morphingTarget(cfg)
# mu = (ceil+floor)/2
# sig = (ceil-floor)/2.3548
# print(f"floor={floor}, ceil={ceil}")
# print(f"mu={mu}, sig={sig}")
# np.save(f"{cfg.feedback_dir}morphingTarget",[mu,sig])
# [mu,sig]=np.load(f"{cfg.feedback_dir}morphingTarget.npy")
# y=gaussian(x, mu, sig)
# plt.plot(x,y)













# # 
# # volume and save that in day2 in day1 space.


# # This script also process the two day 2 recognition run and get the metric floor and ceil (in practice calculate the sigma and mu of the metric transformation function)
# # The relationship between floor, ceil  and mu, sigma is 

# ########################################################
# ########################################################

# # floor=1
# # ceil=-1

# # mu = (floor+ceil)/2
# # sig = (floor-ceil)/2.3548

# # x=np.arange(-3,3,0.01)
# # y=gaussian(x, mu, sig)

# # plt.plot(x,y)

# ########################################################
# ########################################################

# # The way to calculate floor and ceil is 
# # Floor is C evidence for CD classifier (can also be D evidence for CD classifier, they are effectively the same)
# # Ceil is A evidence in AC and AD classifier.

# ########################################################
# ########################################################

# import os
# import sys
# import argparse
# import numpy as np
# import nibabel as nib
# import scipy.io as sio
# from subprocess import call
# from nibabel.nicom import dicomreaders
# import pydicom as dicom  # type: ignore
# import time
# from glob import glob
# import shutil
# from nilearn.image import new_img_like
# import joblib
# sys.path.append('/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/')
# import rtCommon.utils as utils
# from rtCommon.utils import loadConfigFile
# from rtCommon.fileClient import FileInterface
# import rtCommon.projectUtils as projUtils
# from rtCommon.imageHandling import readRetryDicomFromFileInterface, getDicomFileName, convertDicomImgToNifti

# argParser = argparse.ArgumentParser()
# argParser.add_argument('--config', '-c', default='pilot_sub001.ses2.toml', type=str, help='experiment file (.json or .toml)')
# args = argParser.parse_args()
# from rtCommon.cfg_loading import mkdir,cfg_loading
# cfg = utils.loadConfigFile(args.config)

# realtime_ornt=nib.orientations.axcodes2ornt(('I', 'P', 'L'))
# ref_ornt=nib.orientations.axcodes2ornt(('P', 'S', 'L'))
# global ornt_transform
# ornt_transform = nib.orientations.ornt_transform(realtime_ornt,ref_ornt)


# YYYYMMDD= cfg.YYYYMMDD #'20201009' '20201015'
# LASTNAME=cfg.realtimeFolder_subjectName
# PATIENTID=cfg.realtimeFolder_subjectName
# tmp_folder=f'/gpfs/milgram/scratch60/turk-browne/kp578/{YYYYMMDD}.{LASTNAME}.{PATIENTID}/'
# mkdir(tmp_folder)

# subjectFolder = cfg.dicom_dir #: the folder for current subject # For each patient, a new folder will be generated:

# dicomFiles=glob(f"{subjectFolder}*")
# day2templateVolume_dicom=dicomFiles[int(len(dicomFiles)/2)]
# day1templateFunctionalVolume='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/pilot_sub001/ses1_recognition/run1/nifti/templateFunctionalVolume.nii.gz'
# day2templateVolume_fileName='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/pilot_sub001/ses2_recognition/templateFunctionalVolume.nii.gz'
# day2templateVolume_nii=dicom2nii(day2templateVolume_dicom, day2templateVolume_fileName,day1templateFunctionalVolume) # convert dicom to nifti
# # templateVolume=dicom2nii(templateVolume)

# main_folder='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/'
# day2functionalTemplate=f"{main_folder}subjects/{sub}/ses2_recognition/functionalTemplate.nii.gz"
# call(f"cp {day2templateVolume_nii} {day2functionalTemplate}",shell=True)

# day2functionalTemplate_inDay1=f"{main_folder}subjects/{sub}/ses2_feedback/day2functionalTemplate_inDay1.nii.gz"
# command = f'flirt -in {day2functionalTemplate} \
# -ref {day1templateFunctionalVolume} \
# -out {day2functionalTemplate_inDay1}' 

# print(command)
# call(command,shell=True)


# ########################################################
# ########################################################

# # floor=1
# # ceil=-1

# # mu = (floor+ceil)/2
# # sig = (floor-ceil)/2.3548

# # x=np.arange(-3,3,0.01)
# # y=gaussian(x, mu, sig)

# # plt.plot(x,y)

# ########################################################
# ########################################################

# # The way to calculate floor and ceil is 
# # Floor is C evidence for CD classifier (can also be D evidence for CD classifier, they are effectively the same)
# # Ceil is A evidence in AC and AD classifier.


# ## - load the saved model and apply it on the new coming dicom file.
# model_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/clf/'

# # A : bed
# # B : chair
# # C : bench
# # D : table

# # AC_clf = joblib.load(model_dir+'pilot_sub001_benchtable_tablebed.joblib') 
# # AD_clf = joblib.load(model_dir+'pilot_sub001_benchtable_tablechair.joblib')
# AC_clf 
# AD_clf
# CD_clf


# def gaussian(x, mu, sig):
#     # mu and sig is determined before each neurofeedback session using 2 recognition runs.
#     return round(20*(1 - np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))))



# A = "bed"
# B = "chair"
# C = "bench"
# D = "table"

# # then do this for each TR
# # s1 = clf1.score(newTR, ['table'])
# # s2 = clf2.score(newTR, ['table'])

# # In the case of 2 recognition runs, does the floor definition still makes sense? C evidence for CD classifier when A is presented. 1). this is not set to be small, can at times be large. 2). this is not using all data available in these two recognition runs.
# floor=CD_clf.score()
# ceil=-1

# mu = (floor+ceil)/2
# sig = (floor-ceil)/2.3548

# x=np.arange(-3,3,0.01)
# y=gaussian(x, mu, sig)














# # def dicom2nii(templateVolume, filename,templateFunctionalVolume):
# #     dicomObject = dicom.read_file(templateVolume)
# #     niftiObject = dicomreaders.mosaic_to_nii(dicomObject)
# #     # print(nib.aff2axcodes(niftiObject.affine))
# #     temp_data = niftiObject.get_data()
# #     output_image_correct = nib.orientations.apply_orientation(temp_data, ornt_transform)
# #     correct_object = new_img_like(templateFunctionalVolume, output_image_correct, copy_header=True)
# #     print(nib.aff2axcodes(correct_object.affine))
# #     splitList=filename.split('/')
# #     # fullNiftiFilename="/".join(splitList[0:-1])+'/'+splitList[-1].split('.')[0]+'.nii.gz'
# #     fullNiftiFilename=os.path.join(tmp_folder, splitList[-1].split('.')[0]+'.nii.gz')
# #     print('fullNiftiFilename=',fullNiftiFilename)
# #     correct_object.to_filename(fullNiftiFilename)
# #     return fullNiftiFilename


# # fetch the data folder
# # tomlFIle=f"/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/conf/tProject.toml"

# # YYYYMMDD= '20201009' #'20201009' '20201015'
# # YYYYMMDD= '20201019' #'20201009' '20201015'
# # LASTNAME='rtSynth_pilot001'
# # PATIENTID='rtSynth_pilot001'

# # Top_directory = '/gpfs/milgram/project/realtime/DICOM/'
# # sub='pilot_sub001'