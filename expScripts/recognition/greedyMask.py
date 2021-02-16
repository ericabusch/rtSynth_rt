# def greedyMask(cfg):
#     return 0



'''
purpose:
    starting from 31 ROIs, get the best performed ROI combination in a greedy way
    this code is aggregate_greedy.py adapted to match rtcloud
steps:
    load the 31 ROIs from result of neurosketch dataset
    train the model using the 31ROIs and get the accuracy.

    get the N combinations of N-1 ROIs
    retrain the model and get the accuracy for these N combinations

    get the N-1 combinations of N-2 ROIs
    retrain the model and get the accuracy for these N-1 combinations

    when everything is finished, find the best ROI and save as cfg.chosenMask
    
'''
import os
print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}") 
import numpy as np
import nibabel as nib
import sys
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
import itertools
# from tqdm import tqdm
import pickle
import subprocess
from subprocess import call
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# What subject are you running
'''
Takes args (in order):
    subject (e.g. sub001)
    dataSource (e.g. realtime)
    roiloc (wang2014 or schaefer2018)
    N (the number of parcels or ROIs to start with)
'''


from rtCommon.cfg_loading import mkdir,cfg_loading
config="sub001.ses1.toml"
cfg = cfg_loading(config)





subject,dataSource,roiloc,N=cfg.subjectName,"realtime","schaefer2018",31
# subject,dataSource,roiloc,N=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])

print("Running subject {}, with {} as a data source, {}, starting with {} ROIs".format(subject, dataSource, roiloc, N))


# dataSource depending, there are a number of keywords to fill in: 
# ses: which day of data collection
# run: which run number on that day (single digit)
# phase: 12, 34, or 56
# sub: subject number
if dataSource == "neurosketch":
    funcdata = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}_neurosketch/data/nifti/realtime_preprocessed/{sub}_neurosketch_recognition_run_{run}.nii.gz"
    metadata = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/data/features/recog/metadata_{sub}_V1_{phase}.csv"
    anat = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}_neurosketch/data/nifti/{sub}_neurosketch_anat_mprage_brain.nii.gz"
elif dataSource == "realtime":
    funcdata = cfg.recognition_dir + "brain_run{run}.npy"
    metadata = cfg.recognition_dir + "behav_run{run}.csv"
    # funcdata = "/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/{sub}/ses{ses}_recognition/run0{run}/nifti/{sub}_functional.nii.gz"
    # metadata = "/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/{sub}/ses{ses}_recognition/run0{run}/{sub}_0{run}_preprocessed_behavData.csv"
    # anat = "$TO_BE_FILLED"
else:
    funcdata = "/gpfs/milgram/project/turk-browne/projects/rtTest/searchout/feat/{sub}_pre.nii.gz"
    metadata = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/data/features/recog/metadata_{sub}_V1_{phase}.csv"
    anat = "$TO_BE_FILLED"


workingDir="/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/sub001/ses1/recognition"
# workingDir="/gpfs/milgram/project/turk-browne/projects/rtTest/"
# subjects_correctly_aligned=['1206161','0119173','1206162','1130161','1206163','0120171','0111171','1202161','0125172','0110172','0123173','0120173','0110171','0119172','0124171','0123171','1203161','0118172','0118171','0112171','1207162','0117171','0119174','0112173','0112172']
# if roiloc == "schaefer2018":
#     RESULT=np.empty((len(subjects_correctly_aligned),300))
#     topN = []
#     for ii,sub in enumerate(subjects_correctly_aligned):
#         outloc = workingDir+"/{}/{}/output".format(roiloc, sub)
#         for roinum in range(1,301):
#             result = np.load("{}/{}.npy".format(outloc, roinum))
#             RESULT[ii,roinum-1]=result
#             # RESULT = result if roinum == 1 else np.vstack((RESULT, result))
#     RESULT = np.mean(RESULT,axis=0)
#     print(f"RESULT.shape={RESULT.shape}")
#     RESULTix = RESULT[:].argsort()[-N:]
#     for idx in RESULTix:
#         topN.append("{}.nii.gz".format(idx+1))
#         # print(topN[-1])
# else:
#     RESULT_all=[]
#     topN = []
#     for ii,sub in enumerate(subjects_correctly_aligned):
#         outloc = workingDir+"/{}/{}/output".format(roiloc, sub)
#         for hemi in ["lh", "rh"]:
#             for roinum in range(1, 26):
#                 result = np.load("{}/roi{}_{}.npy".format(outloc, roinum, hemi))
#                 Result = result if roinum == 1 else np.vstack((Result, result))
#             RESULT = Result if hemi == "lh" else np.hstack((RESULT, Result))
#         RESULT_all.append(RESULT)

#     RESULT_all=np.asarray(RESULT_all)
#     print(f"RESULT_all.shape={RESULT_all.shape}")
#     RESULT_all=np.mean(RESULT_all,axis=0)
#     print(f"RESULT_all.shape={RESULT_all.shape}")
#     RESULT1d = RESULT.flatten()
#     RESULTix = RESULT1d.argsort()[-N:]
#     x_idx, y_idx = np.unravel_index(RESULTix, RESULT.shape)

#     # Check that we got the largest values.
#     for x, y, in zip(x_idx, y_idx):
#         print(x,y)
#         if y == 0:
#             topN.append("roi{}_lh.nii.gz".format(x+1))
#         else:
#             topN.append("roi{}_rh.nii.gz".format(x+1))
#         # print(topN[-1])

topN = load_obj(f"{cfg.recognition_expScripts_dir}top31ROIs")
print(f"len(topN)={len(topN)}")
print(f"topN={topN}")

def Wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))
        
def normalize(X):
    X = X - X.mean(3)
    return X


# phasedict = dict(zip([1,2,3,4,5,6，7，8],["12", "12", "34", "34", "56", "56"]))
imcodeDict={"A": "bed", "B": "Chair", "C": "table", "D": "bench"}

def getMask(topN, cfg):
    for pn, parc in enumerate(topN):
        _mask = nib.load(cfg.recognition_dir+"mask/schaefer_{}".format(parc))
        # schaefer_56.nii.gz
        aff = _mask.affine
        _mask = _mask.get_data()
        _mask = _mask.astype(int)
        # say some things about the mask.
        mask = _mask if pn == 0 else mask + _mask
        mask[mask>0] = 1
    return mask

mask=getMask(topN, cfg)

print('mask dimensions: {}'. format(mask.shape))
print('number of voxels in mask: {}'.format(np.sum(mask)))


runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])]) # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]
if len(actualRuns) < 8:
    runRecording_preDay = pd.read_csv(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/../runRecording.csv")
    actualRuns_preDay = list(runRecording_preDay['run'].iloc[list(np.where(1==1*(runRecording_preDay['type']=='recognition'))[0])])[-(8-len(actualRuns)):] # might be [5,6,7,8]
else: 
    actualRuns_preDay = []

assert len(actualRuns_preDay)+len(actualRuns)==8 

objects = ['bed', 'bench', 'chair', 'table']

brain_data=[]
behav_data=[]
for ii,run in enumerate(actualRuns): # load behavior and brain data for current session
    t = np.load(f"{cfg.recognition_dir}brain_run{run}.npy")
    brain_data.append(t)

    t = pd.read_csv(f"{cfg.recognition_dir}behav_run{run}.csv")
    t=list(t['Item'])
    behav_data.append(t)

for ii,run in enumerate(actualRuns_preDay): # load behavior and brain data for previous session
    t = np.load(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/brain_run{run}.npy")
    brain_data.append(t)

    t = pd.read_csv(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/behav_run{run}.csv")
    t=list(t['Item'])
    behav_data.append(t)

save_obj([brain_data,behav_data],f"./tmp_folder/{subject}_{dataSource}_{roiloc}_{N}") #{len(topN)}_{i}
# bcvar = [behav_data]
# runs = brain_data
# save_obj([bcvar,runs],f"./tmp_folder/{subject}_{dataSource}_{roiloc}_{N}") #{len(topN)}_{i}

# # Compile preprocessed data and corresponding indices
# metas = []
# for run in range(1, 8):
#     print(run, end='--')
#     # retrieve from the dictionary which phase it is, assign the session
#     phase = phasedict[run]
#     ses = 1
    
#     # Build the path for the preprocessed functional data
#     this4d = funcdata.format(ses=ses, run=run, phase=phase, sub=subject)
    
#     # Read in the metadata, and reduce it to only the TR values from this run, add to a list
#     thismeta = pd.read_csv(metadata.format(ses=ses, run=run, phase=phase, sub=subject))
#     if dataSource == "neurosketch":
#         _run = 1 if run % 2 == 0 else 2
#     else:
#         _run = run
#     thismeta = thismeta[thismeta['run_num'] == int(_run)]
    
#     if dataSource == "realtime":
#         TR_num = list(thismeta.TR.astype(int))
#         labels = list(thismeta.Item)
#         labels = [imcodeDict[label] for label in labels]
#     else:
#         TR_num = list(thismeta.TR_num.astype(int))
#         labels = list(thismeta.label)
    
#     print("LENGTH OF TR: {}".format(len(TR_num)))
#     # Load the functional data
#     runIm = nib.load(this4d)
#     affine_mat = runIm.affine
#     runImDat = runIm.get_data()
    
#     # Use the TR numbers to select the correct features
#     features = [runImDat[:,:,:,n+3] for n in TR_num]
#     features = np.array(features)
#     # features = features[:, mask==1]
#     print("shape of features", features.shape, "shape of mask", mask.shape)
#     featmean = features.mean(1).mean(1).mean(1)[..., None,None,None] #features.mean(1)[..., None]
#     features = features - featmean
#     features = np.expand_dims(features, 0)
    
#     # Append both so we can use it later
#     metas.append(labels)
#     runs = features if run == 1 else np.concatenate((runs, features))

# dimsize = runIm.header.get_zooms()
# # Preset the variables
# print("Runs shape", runs.shape)
# bcvar = [metas]


                 
# # Distribute the information to the searchlights (preparing it to run)
# _runs = [runs[:,:,mask==1]]
# print("Runs shape", _runs[0].shape)
# slstart = time.time()
# sl_result = Class(_runs, bcvar)
# print("results of classifier: {}, type: {}".format(sl_result, type(sl_result)))
# SL = time.time() - slstart
# tot = time.time() - starttime
# print('total time: {}, searchlight time: {}'.format(tot, SL))

def wait(tmpFile):
    while not os.path.exists(tmpFile+'_result.npy'):
        time.sleep(5)
        print(f"waiting for {tmpFile}_result.npy\n")
    return np.load(tmpFile+'_result.npy')

def numOfRunningJobs():
    # subprocess.Popen(['squeue -u kp578 | wc -l > squeue.txt'],shell=True) # sl_result = Class(_runs, bcvar)
    randomID=str(time.time())
    # print(f"squeue -u kp578 | wc -l > squeue/{randomID}.txt")
    call(f'squeue -u kp578 | wc -l > squeue/{randomID}.txt',shell=True)
    numberOfJobsRunning = int(open(f"squeue/{randomID}.txt", "r").read())
    print(f"numberOfJobsRunning={numberOfJobsRunning}")
    return numberOfJobsRunning



def Class(brain_data,behav_data):
    # metas = bcvar[0]
    # data4d = data[0]
    print([t.shape for t in brain_data])

    accs = []
    for run in range(8):
        testX = brain_data[run]
        testY = behav_data[run]

        trainX=np.zeros((1,1))
        for i in range(8):
            if i !=run:
                trainX=brain_data[i] if trainX.shape==(1,1) else np.concatenate((trainX,brain_data[i]),axis=0)

        trainY = []
        for i in range(8):
            if i != run:
                trainY.extend(behav_data[i])
        clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                 multi_class='multinomial').fit(trainX, trainY)
                
        # Monitor progress by printing accuracy (only useful if you're running a test set)
        acc = clf.score(testX, testY)
        accs.append(acc)
    
    return np.mean(accs)

if not os.path.exists(f"./tmp_folder/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)}.pkl"):
    brain_data = [t[:,mask==1] for t in brain_data]
    # _runs = [runs[:,mask==1]]
    print("Runs shape", [t.shape for t in brain_data])
    slstart = time.time()
    sl_result = Class(brain_data, behav_data)

    save_obj({"subject":subject,
    "startFromN":N,
    "currNumberOfROI":len(topN),
    "bestAcc":sl_result, # this is the sl_result for the topN, not the bestAcc, bestAcc is for the purpose of keeping consistent with others
    "bestROIs":topN},# this is the topN, not the bestROIs, bestROIs is for the purpose of keeping consistent with others
    f"./tmp_folder/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)}"
    )
# ./tmp_folder/0125171_40_schaefer2018_neurosketch_39.pkl
if os.path.exists(f"./tmp_folder/{subject}_{N}_{roiloc}_{dataSource}_{1}.pkl"):
    print(f"./tmp_folder/{subject}_{N}_{roiloc}_{dataSource}_1.pkl exists")
    raise Exception('runned or running')

# N-1
def next(topN):
    print(f"len(topN)={len(topN)}")
    print(f"topN={topN}")

    if len(topN)==1:
        return None
    else:
        try:
            allpairs = itertools.combinations(topN,len(topN)-1)
            topNs=[]
            sl_results=[]
            tmpFiles=[]
            while os.path.exists("./tmp_folder/holdon.npy"):
                time.sleep(10)
                print("sleep for 10s ; waiting for ./tmp_folder/holdon.npy to be deleted")
            np.save("./tmp_folder/holdon",1)

            for i,_topN in enumerate(allpairs):
                tmpFile=f"./tmp_folder/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)}_{i}"
                print(f"tmpFile={tmpFile}")
                topNs.append(_topN)
                tmpFiles.append(tmpFile)

                if not os.path.exists(tmpFile+'_result.npy'):
                    # prepare brain data(runs) mask and behavior data(bcvar) 

                    save_obj([_topN,subject,dataSource,roiloc,N], tmpFile)

                    print("kp2")
                    numberOfJobsRunning = numOfRunningJobs()
                    print("kp3")
                    while numberOfJobsRunning > 400: # 300 is not filling it up
                        print("kp4 300")
                        print("waiting 10, too many jobs running") ; time.sleep(10)
                        numberOfJobsRunning = numOfRunningJobs()
                        print("kp5")

                    # get the evidence for the current mask
                    print(f'sbatch class.sh {tmpFile}')
                    proc = subprocess.Popen([f'sbatch --requeue class.sh {tmpFile}'],shell=True) # sl_result = Class(_runs, bcvar) 
                    print("kp6")
                else:
                    print(tmpFile+'_result.npy exists!')
            os.remove("./tmp_folder/holdon.npy")

            # wait for everything to be finished and make a summary to find the best performed megaROI
            sl_results=[]
            for tmpFile in tmpFiles:
                sl_result=wait(tmpFile)
                sl_results.append(sl_result)
            print(f"sl_results={sl_results}")
            print(f"max(sl_results)=={max(sl_results)}")
            maxID=np.where(sl_results==max(sl_results))[0][0]
            save_obj({"subject":subject,
            "startFromN":N,
            "currNumberOfROI":len(topN)-1,
            "bestAcc":max(sl_results),
            "bestROIs":topNs[maxID]},
            f"./tmp_folder/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)-1}"
            )
            print(f"bestAcc={max(sl_results)} For {len(topN)-1} = ./tmp_folder/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)-1}")
            tmpFiles=next(topNs[maxID])
        except:
            return tmpFiles
tmpFiles=next(topN)


