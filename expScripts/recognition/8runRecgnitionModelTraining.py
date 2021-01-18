#  this script is meant to deal with the data of 8 recognition runs and generate models saved in corresponding folder
'''
input:
    cfg.session=ses1
    cfg.modelFolder=f"{cfg.subjects_dir}/{cfg.subjectName}/{cfg.session}_recognition/clf/"
    cfg.dataFolder=f"{cfg.subjects_dir}/{cfg.subjectName}/{cfg.session}_recognition/"
output:
    models in cfg.modelFolder
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
argParser.add_argument('--config', '-c', default='sub001.ses1.toml', type=str, help='experiment file (.json or .toml)')
args = argParser.parse_args()
from rtCommon.cfg_loading import mkdir,cfg_loading
cfg = cfg_loading(args.config)

sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/')
from recognition_dataAnalysisFunctions import recognition_preprocess,minimalClass,behaviorDataLoading


'''
convert all dicom files into nii files in the temp dir. 
find the middle volume of the run1 as the template volume
align every other functional volume with templateFunctionalVolume (3dvolreg)
'''
recognition_preprocess(cfg)

'''
run the mask selection
    make ROIs
        make-wang-rois.sh
        make-schaefer-rois.sh
    train classifiers on each ROI
    summarize classification accuracy and select best mask
'''
# make ROIs
    # make-wang-rois.sh
call(f"sbatch {cfg.recognition_expScripts_dir}make-wang-rois.sh {cfg.subjectName} {cfg.recognition_dir}")
    # make-schaefer-rois.sh
call(f"sbatch {cfg.recognition_expScripts_dir}make-schaefer-rois.sh {cfg.subjectName} {cfg.recognition_dir}")

# train classifiers on each ROI
call(f"sbatch {cfg.recognition_expScripts_dir}batchRegions.sh {args.config}")

# summarize classification accuracy and select best mask
call(f"sbatch {cfg.recognition_expScripts_dir}aggregate.sh {cfg.subjectName} {cfg.recognition_dir}")

# select the mask with the best performance as cfg.chosenMask = {cfg.recognition_dir}chosenMask.nii.gz
# and also save this mask in all 

'''
load preprocessed and aligned behavior and brain data 
select data with the wanted pattern like AB AC AD BC BD CD 
train correspondng classifier and save the classifier performance and the classifiers themselves.
'''
minimalClass(cfg)


