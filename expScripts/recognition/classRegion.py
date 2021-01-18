'''
This script is adapted from classRegion.py
Purpose:
    to train and save the classifiers for all ROIs

'''
 

'''
from the recognition exp dir, run batchRegions.sh, it will run the script classRegion.sh, which is just a feeder for classRegion.py for all ROI/parcels across both wang and schaefer.

classRegion.py simply runs a runwise cross-validated classifier across the runs of recognition data, then stores the average accuracy of the ROI it was assigned in an numpy array. 
This is stored within the subject specific folder (e.g. wang2014/0111171/output/roi25_rh.npy )

input:
    1 subject: which subject
    2 dataloc: neurosketch or realtime
    3 roiloc: schaefer or wang
    4 roinum: number of rois you want
    5 roihemi: which hemisphere

'''


import os
import sys
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')
import argparse
import numpy as np
import nibabel as nib
import scipy.io as sio
from subprocess import call
import pandas as pd
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

toml=sys.argv[1]
from rtCommon.cfg_loading import mkdir,cfg_loading
cfg = cfg_loading(toml)

runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
cfg.actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])])


import nibabel as nib
import numpy as np
import os
import sys
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression

# What subject are you running
dataSource = "realtime"

try:
    roiloc = str(sys.argv[3])
    print("Using user-selected roi location: {}".format(roiloc))
except:
    print("NO ROI LOCATION ENTERED: Using radius of wang2014")
    roiloc = "wang"

try:
    dataSource = sys.argv[2]  # could be neurosketch or realtime
    print("Using {} data".format(dataSource))
except:
    print("NO DATASOURCE ENTERED: Using original neurosketch data")
    dataSource = 'neurosketch'

try:
    roinum = str(sys.argv[4]) if roiloc == "schaefer2018" else "roi{}".format(str(sys.argv[4])) 
    print("running for roi #{} in {}".format(str(sys.argv[4]), roiloc))
except:
    print("NO ROI SPECIFIED: Using roi number 1")
    roinum="1"

if roiloc == "wang":
    try:
        roihemi = "_{}".format(str(sys.argv[5]))
        print("Since this is wang2014, we need a hemisphere, in this case {}".format(str(sys.argv[5])))
    except:
        print("this is wang 2014, so we need a hemisphere, but one was not specified")
        assert 1 == 2
else:
    roihemi=""

print("Running subject {}, with {} as a data source, {} roi #{} {}".format(cfg.subjectName, dataSource, roiloc, roinum, roihemi))

starttime = time.time()

def Wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))
        
def normalize(X):
    X = X - X.mean(3)
    return X

def Class(data, bcvar):
    metas = bcvar
    data4d = data
    accs = []
    for curr_run in range(8):
        testX = data4d[curr_run]
        testY = metas[curr_run]
        trainX=None
        for train_run in range(8):
            if train_run!=curr_run:
                trainX = data4d[train_run] if type(trainX)!=np.ndarray else np.concatenate((trainX, data4d[train_run]),axis=0)
        trainY = []
        for train_run in range(8):
            if train_run!=curr_run:
                trainY.extend(metas[train_run])
        # remove nan type
        id=[type(i)==str for i in trainY]
        trainY=[i for i in trainY if type(i)==str]
        trainX=trainX[id]

        clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                 multi_class='multinomial').fit(trainX, trainY)

        # Monitor progress by printing accuracy (only useful if you're running a test set)
        id=[type(i)==str for i in testY]
        testY=[i for i in testY if type(i)==str]
        testX=testX[id]
        acc = clf.score(testX, testY)
        accs.append(acc)
    return np.mean(accs)

phasedict = dict(zip([1,2,3,4,5,6,7,8],[cfg.actualRuns]))
imcodeDict={"A": "bed", "B": "Chair", "C": "table", "D": "bench"}

mask = nib.load(f"{cfg.mask_dir}{roiloc}_{roinum}{roihemi}.nii.gz").get_data()
mask = mask.astype(int)
# say some things about the mask.
print('mask dimensions: {}'. format(mask.shape))
print('number of voxels in mask: {}'.format(np.sum(mask)))


# Compile preprocessed data and corresponding indices
metas = []
runs=[]
for run_i,run in enumerate(cfg.actualRuns):
    print(run, end='--')
    
    # Build the path for the preprocessed functional data
    this4d = f"{cfg.recognition_dir}run{run}.nii.gz" # run data
    
    # Read in the metadata, and reduce it to only the TR values from this run, add to a list
    thismeta = pd.read_csv(f"{cfg.recognition_dir}{cfg.subjectName}_{run_i+1}.csv")

    TR_num = list(thismeta.TR.astype(int))
    labels = list(thismeta.Item)
    labels = [None if type(label)==float else imcodeDict[label] for label in labels]


    print("LENGTH OF TR: {}".format(len(TR_num)))
    # Load the functional data
    runIm = nib.load(this4d)
    affine_mat = runIm.affine
    runImDat = runIm.get_data()
    
    # Use the TR numbers to select the correct features
    features = [runImDat[:,:,:,n+2] for n in TR_num]
    features = np.array(features)
    features = features[:, mask==1]
    print("shape of features", features.shape, "shape of mask", mask.shape)
    featmean = features.mean(1)[..., None]
    features = features - featmean
    # features = np.expand_dims(features, 0)
    
    # Append both so we can use it later
    metas.append(labels)
    runs.append(features) # if run_i == 0 else np.concatenate((runs, features))


dimsize = runIm.header.get_zooms()


# Preset the variables
print("len(Runs)", len(runs))
data = runs
print("Runs shape",[i.shape for i in data])
bcvar = metas
                 
# Distribute the information to the searchlights (preparing it to run)
slstart = time.time()
sl_result = Class(data, bcvar)
print("results of classifier: {}, type: {}".format(sl_result, type(sl_result)))

SL = time.time() - slstart
tot = time.time() - starttime
print('total time: {}, searchlight time: {}'.format(tot, SL))

mkdir(f"{cfg.recognition_dir}classRegions/")
outfile = f"{cfg.recognition_dir}classRegions/{roiloc}_{roinum}_{roihemi}.npy"
print(outfile)
np.save(outfile, np.array(sl_result))
