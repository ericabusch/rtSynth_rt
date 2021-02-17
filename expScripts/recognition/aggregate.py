'''
steps:
    load best performed ROI masks
    for given number of ROIs wanted, combine them and use that as the wanted mask
    train a new classifier and save the testing accuracy
'''


'''
you could try to see whether combining parcels improves performance. 
That's going to be the most important bit, because we'll want to decide on a tradeoff between number of voxels and accuracy. 
The script of interest here is aggregate.sh which is just a feeder for aggregate.py. 
This will use the .npy outputs of classRegion.py to select and merge the top N ROIs/parcels, and will return the list of ROI names, the number of voxels, and the cross-validated classifier accuracy 
in this newly combined larger mask. An example run of this is as follows:
sbatch aggregate.sh 0111171 neurosketch schaefer2018 15
'''

# test: sbatch /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/aggregate.sh sub001.ses1.toml realtime schaefer 299

import os
print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}")
# /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtcloud
# /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud
import numpy as np
import nibabel as nib
import sys
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')
import argparse
import scipy.io as sio
from subprocess import call
from nibabel.nicom import dicomreaders
import pydicom as dicom  # type: ignore
from glob import glob
import shutil
from nilearn.image import new_img_like
# import rtCommon.utils as utils
# from rtCommon.utils import loadConfigFile
# from rtCommon.fileClient import FileInterface
# import rtCommon.projectUtils as projUtils
# from rtCommon.imageHandling import readRetryDicomFromFileInterface, getDicomFileName, convertDicomImgToNifti
from rtCommon.cfg_loading import mkdir,cfg_loading


'''
Takes args (in order):
    toml
    dataSource (e.g. neurosketch, but also realtime)
    roiloc (wang or schaefer)
    N (the number of parcels or ROIs to start with)
'''
toml = sys.argv[1] # sub001.ses1.toml
cfg = cfg_loading(toml) 
N = int(sys.argv[4]) # 20
roiloc = str(sys.argv[3]) #wang or schaefer 
print("Using user-selected roi location: {}".format(roiloc))
dataSource = sys.argv[2]  # could be neurosketch or realtime
print("Using {} data".format(dataSource))
print("Running subject {}, with {} as a data source, {}, starting with {} ROIs".format(cfg.subjectName, dataSource, roiloc, N))


# if dataSource == "neurosketch":
#     funcdata = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}_neurosketch/data/nifti/realtime_preprocessed/{sub}_neurosketch_recognition_run_{run}.nii.gz"
#     metadata = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/data/features/recog/metadata_{sub}_V1_{phase}.csv"
#     anat = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}_neurosketch/data/nifti/{sub}_neurosketch_anat_mprage_brain.nii.gz"
# elif dataSource == "realtime":
#     funcdata = "/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/{sub}/ses{ses}_recognition/run0{run}/nifti/{sub}_functional.nii.gz"
#     metadata = "/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/{sub}/ses{ses}_recognition/run0{run}/{sub}_0{run}_preprocessed_behavData.csv"
#     anat = "$TO_BE_FILLED"
# else:
#     funcdata = "/gpfs/milgram/project/turk-browne/projects/rtTest/searchout/feat/{sub}_pre.nii.gz"
#     metadata = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/data/features/recog/metadata_{sub}_V1_{phase}.csv"
#     anat = "$TO_BE_FILLED"
    
starttime = time.time()

outloc = f'{cfg.recognition_dir}classRegions/'


if roiloc == "schaefer":
    topN = []
    for roinum in range(1,301):
        result = np.load(f"{outloc}/{roiloc}_{roinum}_.npy") #this is the saved testing accuracy
        RESULT = result if roinum == 1 else np.vstack((RESULT, result))
    RESULTix = RESULT[:,0].argsort()[-N:]
    for idx in RESULTix:
        topN.append("{}.nii.gz".format(idx+1))
        print(topN[-1])
else:
    topN = []
    for hemi in ["lh", "rh"]:
        for roinum in range(1, 26):
            result = np.load(f"{outloc}{roiloc}_roi{roinum}__{hemi}.npy")
            Result = result if roinum == 1 else np.vstack((Result, result))
        RESULT = Result if hemi == "lh" else np.hstack((RESULT, Result))

    RESULT1d = RESULT.flatten()
    RESULTix = RESULT1d.argsort()[-N:]
    x_idx, y_idx = np.unravel_index(RESULTix, RESULT.shape)

    # Check that we got the largest values.
    for x, y, in zip(x_idx, y_idx):
        print(x,y)
        if y == 0:
            topN.append("roi{}_lh.nii.gz".format(x+1))
        else:
            topN.append("roi{}_rh.nii.gz".format(x+1))
        print(topN[-1])


def Wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))
        
def normalize(X):
    X = X - X.mean(3)
    return X

def Class(data, bcvar):
    metas = bcvar[0]
    data4d = data[0]
    print(data4d.shape)

    accs = []
    for run in range(6):
        testX = data4d[run]
        testY = metas[run]
        trainX = data4d[np.arange(6) != run]
        trainX = trainX.reshape(trainX.shape[0]*trainX.shape[1], -1)
        trainY = []
        for meta in range(6):
            if meta != run:
                trainY.extend(metas[run])
        clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                 multi_class='multinomial').fit(trainX, trainY)
                
        # Monitor progress by printing accuracy (only useful if you're running a test set)
        acc = clf.score(testX, testY)
        accs.append(acc)
    
    return np.mean(accs)


# phasedict = dict(zip([1,2,3,4,5,6],["12", "12", "34", "34", "56", "56"]))
imcodeDict={"A": "bed", "B": "Chair", "C": "table", "D": "bench"}

for pn, parc in enumerate(topN):
    _mask = nib.load(f'{cfg.recognition_dir}mask/{roiloc}_{parc}')
    aff = _mask.affine
    _mask = _mask.get_data()
    _mask = _mask.astype(int)
    # say some things about the mask.
    mask = _mask if pn == 0 else mask + _mask
    mask[mask>0] = 1
print('mask dimensions: {}'. format(mask.shape))
print('number of voxels in mask: {}'.format(np.sum(mask)))

runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
cfg.actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])])

# # Compile preprocessed data and corresponding indices
# metas = []
# for run_i,run in enumerate(cfg.actualRuns):
#     print(run, end='--')
#     # retrieve from the dictionary which phase it is, assign the session
#     # phase = phasedict[run]
#     # ses = 1
    
#     # Build the path for the preprocessed functional data
#     # this4d = funcdata.format(ses=cfg.session, run=run, sub=cfg.subjectName)
#     this4d = f"{cfg.recognition_dir}run{run}.nii.gz" # run data

#     # Read in the metadata, and reduce it to only the TR values from this run, add to a list
#     # thismeta = pd.read_csv(metadata.format(ses=ses, run=run, phase=phase, sub=subject))
#     # thismeta = pd.read_csv(f"{cfg.recognition_dir}{cfg.subjectName}_{run_i+1}.csv")
#     thismeta = pd.read_csv(f"{cfg.recognition_dir}behav_run{run}.csv")
    
#     # thismeta = thismeta[thismeta['run_num'] == int(run)]

#     TR_num = list(thismeta.TR.astype(int))
#     labels = list(thismeta.Item)
#     labels = [imcodeDict[label] for label in labels]
    
#     print("LENGTH OF TR: {}".format(len(TR_num)))
#     # Load the functional data
#     runIm = nib.load(this4d)
#     affine_mat = runIm.affine
#     runImDat = runIm.get_data()
    
#     # Use the TR numbers to select the correct features
#     features = [runImDat[:,:,:,n+3] for n in TR_num]
#     features = np.array(features)
#     features = features[:, mask==1]
#     print("shape of features", features.shape, "shape of mask", mask.shape)
#     featmean = features.mean(1)[..., None]
#     features = features - featmean
#     features = np.expand_dims(features, 0)
    
#     # Append both so we can use it later
#     metas.append(labels)
#     runs = features if run_i == 0 else np.concatenate((runs, features))

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
    # featmean = features.mean(1)[..., None]
    # features = features - featmean
    features = features - features.mean(0)
    
    # Append both so we can use it later
    metas.append(labels)
    runs.append(features) # if run_i == 0 else np.concatenate((runs, features))


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
data = runs
bcvar = metas
slstart = time.time()
sl_result=Class(data, bcvar)

# dimsize = runIm.header.get_zooms()

# data = []
# # Preset the variables
# print("Runs shape", runs.shape)
# _data = runs
# print(_data.shape)
# data.append(_data)
# print("shape of data: {}".format(_data.shape))

# bcvar = [metas]

# # Distribute the information to the searchlights (preparing it to run)
# slstart = time.time()
# sl_result = Class(data, bcvar)
print("results of classifier: {}, type: {}".format(sl_result, type(sl_result)))

SL = time.time() - slstart
tot = time.time() - starttime
print('total time: {}, searchlight time: {}'.format(tot, SL))

#SAVE accuracy

outfile = f"{cfg.recognition_dir}classRegions/{roiloc}_top{N}.npy"
np.save(outfile, np.array(sl_result))
#SAVE mask
savemask = nib.Nifti1Image(mask, affine=aff)
nib.save(savemask, f"{cfg.recognition_dir}classRegions/{roiloc}_top{N}mask.nii.gz")
#SAVE roilist, nvox
ROILIST = [r for r in topN]
ROILIST.append(np.sum(mask))
ROILIST = pd.DataFrame(ROILIST)
ROILIST.to_csv(f"{cfg.recognition_dir}classRegions/{roiloc}_top{N}.csv")





def plot():
        
    import sys
    sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')
    from rtCommon.cfg_loading import mkdir,cfg_loading
    from glob import glob

    toml="sub001.ses1.toml"
    cfg = cfg_loading(toml) 
    subjects=[cfg.subjectName]

    # testDir='/gpfs/milgram/project/turk-browne/projects/rtTest/'
    hemis=["lh", "rh"]

    wangAcc=np.zeros((50,len(subjects)))
    roiloc="wang"
    for sub_i,sub in enumerate(subjects):
        for num in range(1,51):
            try:
                wangAcc[num-1,sub_i]=np.load(f"{cfg.recognition_dir}classRegions/{roiloc}_top{num}.npy")
            except:
                pass

    schaeferAcc=np.zeros((300,len(subjects)))
    roiloc="schaefer"
    for sub_i,sub in enumerate(subjects):
        for num in range(1,301):
            try:
                schaeferAcc[num-1,sub_i]=np.load(f"{cfg.recognition_dir}classRegions/{roiloc}_top{num}.npy")
            except:
                pass


    wangAcc=wangAcc[:,wangAcc[0]!=0]
    schaeferAcc=schaeferAcc[:,schaeferAcc[0]!=0]
    schaeferAcc[schaeferAcc==0]=None

    import matplotlib.pyplot as plt
    plt.plot(np.nanmean(wangAcc,axis=1))
    plt.plot(np.nanmean(schaeferAcc,axis=1))


    for i in range(schaeferAcc.shape[0]):
        plt.scatter([i]*schaeferAcc.shape[1],schaeferAcc[i],c='g')
    for i in range(wangAcc.shape[0]):
        plt.scatter([i]*wangAcc.shape[1],wangAcc[i],c='b')

    plt.xlabel("number of ROIs")
    plt.ylabel("accuracy")

    bestN=np.where(schaeferAcc==max(schaeferAcc))[0][0]+1
    print(f"best performed ROI N={bestN}")
    print(f"fslview_deprecated {cfg.recognition_dir}wanginfunc.nii.gz \
        {cfg.recognition_dir}classRegions/wang_top{bestN}mask.nii.gz")

    # from shutil import copyfile
    # copyfile(f"{cfg.recognition_dir}classRegions/wang_top{bestN}mask.nii.gz", 
    #         cfg.chosenMask
    #         )

    
    