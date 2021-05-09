import os
import sys
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')
import argparse
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import scipy.io as sio
import subprocess
from scipy.stats import zscore
from nibabel.nicom import dicomreaders
import pydicom as dicom  # type: ignore
import time
from glob import glob
import shutil
from nilearn.image import new_img_like
import joblib
import rtCommon.utils as utils
from rtCommon.utils import loadConfigFile
import pickle5 as pickle
# import and set up environment
import sys
from subprocess import call
import nibabel as nib
import pydicom as dicom
import numpy as np
import time
import os
from glob import glob
import shutil
import pandas as pd
# from import convertDicomFileToNifti
from rtCommon.imageHandling import convertDicomImgToNifti, readDicomFromFile
from rtCommon.cfg_loading import mkdir,cfg_loading

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
# from rtCommon.fileClient import FileInterface
# import rtCommon.projectUtils as projUtils
# from rtCommon.imageHandling import readRetryDicomFromFileInterface, getDicomFileName, convertDicomImgToNifti


argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default='sub001.ses5.toml', type=str, help='experiment file (.json or .toml)')
argParser.add_argument('--skipPre', '-s', default=0, type=int, help='skip preprocess or not')
argParser.add_argument('--skipGreedy', '-g', default=0, type=int, help='skip greedy or not')
args = argParser.parse_args("")
from rtCommon.cfg_loading import mkdir,cfg_loading
# config="sub001.ses2.toml"
cfg = cfg_loading(args.config)


# load points history
for feedbackRun in range(1,7):
    history = pd.read_csv(f"{cfg.feedback_dir}{cfg.subjectName}_{feedbackRun}_history.csv")
    # _=plt.figure()
    _=plt.plot(history['points'],label=str(feedbackRun))
    _=plt.legend()
_=plt.title("points history for each feedback run")


runRecording = pd.read_csv(f"{cfg.feedback_dir}../runRecording.csv")
actualScans = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='feedback'))[0])]) # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]
actualScans



for actualScan in actualScans:
    _=plt.figure()
    _=plt.plot(np.load(f"{cfg.feedback_dir}B_evidences_{actualScan}.npy"))
    _=plt.title(f"B_evidence for scan {actualScan}")

import scipy.io

for actualScan in actualScans:
    _=plt.figure()
    _=plt.plot(scipy.io.loadmat(f"{cfg.feedback_dir}morphParam_{actualScan}.mat")['value'])
    _=plt.title(f"morphParam for scan {actualScan}")