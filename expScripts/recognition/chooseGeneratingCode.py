# choose generating code for every recognition run for the current subject
'''
input: 
    cfg
output:
    choose file for each future recognition run. for day1 2 3 4 5, 8 4 4 4 8 choose numbers are needed.
    After this code is run, before running the recognition run, preRecognition.py 
    will be runned to know the actual number of TR for the current recognition run 
    and input that number into the scanner computer
'''

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
sys.path.append('/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/')
import rtCommon.utils as utils
from rtCommon.utils import loadConfigFile
from rtCommon.fileClient import FileInterface
import rtCommon.projectUtils as projUtils
from rtCommon.imageHandling import readRetryDicomFromFileInterface, getDicomFileName, convertDicomImgToNifti

argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default='pilot_sub001.ses1.toml', type=str, help='experiment file (.json or .toml)')
args = argParser.parse_args()
from rtCommon.cfg_loading import mkdir,cfg_loading
cfg = cfg_loading(args.config)

# def chooseGeneratingCode(cfg):
choose = np.random.choice(np.arange(1, 49), 28, replace=False)

chooseNumbers=[8, 4, 4, 4, 8]
for curr_sess in range(1,6):
    np.save(f"{cfg.subjects_dir}/{cfg.subjectName}/ses{curr_sess}/recognition/choose.npy", 
    choose[
        sum(chooseNumbers[:curr_sess-1]),
        sum(chooseNumbers[:curr_sess])
        ])
        # this part generate 8 4 4 4 8 numbers for ses1 2 3 4 5
        # The following portion of the choose are save seperately to the choose file for each session.
        # 0:8
        # 8:8+4
        # 8+4:8+4+4
        # 8+4+4:8+4+4+4
        # 8+4+4+4:8+4+4+4+8
