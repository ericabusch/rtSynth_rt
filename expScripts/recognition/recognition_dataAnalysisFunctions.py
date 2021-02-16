'''
input: 
    cfg.subjectName
    cfg.dicom_dir # Folder where data is saved 
        # e.g. /gpfs/milgram/project/realtime/DICOM/20201019.rtSynth_pilot001_2.rtSynth_pilot001_2/ 
        # inside which is like 001_000003_000067.dcm

output: save

major steps: 
    figure out the number of runs # should be 8, but need confirmation 
        day1 in cfg.dicom_dir there are 8 runs
        day2 in cfg.dicom_dir there are 2+1+2 runs
        day3 in cfg.dicom_dir there are 2+1+2 runs
        day4 in cfg.dicom_dir there are 2+1+2 runs
        day1 in cfg.dicom_dir there are 8 runs

    # preprocess and alignment
    1. figure out the number of dicoms in each runs, should be the maxTR for each run.
    2. select the middle volume of the first run as the template functional volume and save it inside the cfg and save cfg using pickle
    3. align every other functional volume with templateFunctionalVolume (3dvolreg)
    4. merge the aligned data to the PreprocessedData, finish preprocessing (fslmerge)

    # recog_features.py portion
    5. load the aligned nifti file generated by neurosketch_realtime_preprocess.py
    6. no filter
    7. load mask data and apply mask
    8. load behavior data and push the behavior data back for 2 TRs

    # offlineModelTraining.py portion
    9. load preprocessed and aligned behavior and brain data 
    10. select data with the wanted pattern like AB AC AD BC BD CD 
    11. train correspondng classifier and save the classifier performance and the classifiers themselves.


'''

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

# setting up code testing environment: 
# from rtCommon.cfg_loading import mkdir,cfg_loading ;cfg = cfg_loading('pilot_sub001.ses1.toml')

def recognition_preprocess(cfg): 
    '''
    purpose: 
        prepare data for the model training code.
    steps:
        convert all dicom files into nii files in the temp dir. 
        find the middle volume of the run1 as the template volume
        align every other functional volume with templateFunctionalVolume (3dvolreg)
    '''

    # convert all dicom files into nii files in the temp dir. 
    tmp_dir=f"{cfg.tmp_folder}{time.time()}/" ; mkdir(tmp_dir)
    dicomFiles=glob(f"{cfg.dicom_dir}/*.dcm") ; dicomFiles.sort()
    for curr_dicom in dicomFiles:
        dicomImg = readDicomFromFile(curr_dicom) # read dicom file
        convertDicomImgToNifti(dicomImg, dicomFilename=f"{tmp_dir}/{curr_dicom.split('/')[-1]}") #convert dicom to nii    
        # os.remove(f"{tmp_dir}/{curr_dicom.split('/')[-1]}") # remove temp dcm file

    # find the middle volume of the run1 as the template volume
    tmp=glob(f"{tmp_dir}/001_000001*.nii") ; tmp.sort()
    cfg.templateFunctionalVolume = f"{cfg.recognition_dir}/templateFunctionalVolume.nii" 
    call(f"cp {tmp[int(len(tmp)/2)]} {cfg.templateFunctionalVolume}", shell=True)

    # align every other functional volume with templateFunctionalVolume (3dvolreg)
    allTRs=glob(f"{tmp_dir}/001_*.nii") ; allTRs.sort()

    # select a list of run IDs based on the runRecording.csv, actualRuns would be [1,2] is the 1st and the 3rd runs are recognition runs.
    runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])])
    for curr_run in actualRuns:
        outputFileNames=[]
        runTRs=glob(f"{tmp_dir}/001_{str(curr_run).zfill(6)}_*.nii") ; runTRs.sort()
        for curr_TR in runTRs:
            command = f"3dvolreg \
                -base {cfg.templateFunctionalVolume} \
                -prefix  {curr_TR[0:-4]}_aligned.nii \
                {curr_TR}"
            call(command,shell=True)
            outputFileNames.append(f"{curr_TR[0:-4]}_aligned.nii")
        files=''
        for f in outputFileNames:
            files=files+' '+f
        command=f"fslmerge -t {cfg.recognition_dir}run{curr_run}.nii {files}"
        print('running',command)
        call(command, shell=True)

    # remove the tmp folder
    shutil.rmtree(tmp_dir)

    # load and apply mask
            
    '''
    for each run, 
        load behavior data 
        push the behavior data back for 2 TRs
        save the brain TRs with images
        save the behavior data
    '''

    for curr_run_behav,curr_run in enumerate(actualRuns):
        # load behavior data
        behav_data = behaviorDataLoading(cfg,curr_run_behav+1)

        # brain data is first aligned by pushed back 2TR(4s)
        brain_data = nib.load(f"{cfg.recognition_dir}run{curr_run}.nii.gz").get_data() ; brain_data=np.transpose(brain_data,(3,0,1,2))
        Brain_TR=np.arange(brain_data.shape[0])
        Brain_TR = Brain_TR+2

        # select volumes of brain_data by counting which TR is left in behav_data
        Brain_TR=Brain_TR[list(behav_data['TR'])] # original TR begin with 0
        if Brain_TR[-1]>=brain_data.shape[0]: # when the brain data is not as long as the behavior data, delete the last row
            Brain_TR = Brain_TR[:-1]
            behav_data = behav_data.drop([behav_data.iloc[-1].TR])
        brain_data=brain_data[Brain_TR]
        np.save(f"{cfg.recognition_dir}brain_run{curr_run}.npy", brain_data)
        # save the behavior data
        behav_data.to_csv(f"{cfg.recognition_dir}behav_run{curr_run}.csv")

def minimalClass(cfg):
    '''
    purpose: 
        train offline models

    steps:
        load preprocessed and aligned behavior and brain data 
        select data with the wanted pattern like AB AC AD BC BD CD 
        train correspondng classifier and save the classifier performance and the classifiers themselves.

    '''

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn
    import joblib
    import nibabel as nib
    import itertools
    from sklearn.linear_model import LogisticRegression

    def gaussian(x, mu, sig):
        # mu and sig is determined before each neurofeedback session using 2 recognition runs.
        return round(1+18*(1 - np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))) # map from (0,1) -> [1,19]

    def normalize(X):
        X = X - X.mean(0)
        return X

    def jitter(size,const=0):
        jit = np.random.normal(0+const, 0.05, size)
        X = np.zeros((size))
        X = X + jit
        return X

    def other(target):
        other_objs = [i for i in ['bed', 'bench', 'chair', 'table'] if i not in target]
        return other_objs

    def red_vox(n_vox, prop=0.1):
        return int(np.ceil(n_vox * prop))

    def get_inds(X, Y, pair, testRun=None):
        
        inds = {}
        
        # return relative indices
        if testRun:
            trainIX = Y.index[(Y['label'].isin(pair)) & (Y['run_num'] != int(testRun))]
        else:
            trainIX = Y.index[(Y['label'].isin(pair))]

        # pull training and test data
        trainX = X[trainIX]
        trainY = Y.iloc[trainIX].label
        
        # Main classifier on 5 runs, testing on 6th
        clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                 multi_class='multinomial').fit(trainX, trainY)
        B = clf.coef_[0]  # pull betas

        # retrieve only the first object, then only the second object
        if testRun:
            obj1IX = Y.index[(Y['label'] == pair[0]) & (Y['run_num'] != int(testRun))]
            obj2IX = Y.index[(Y['label'] == pair[1]) & (Y['run_num'] != int(testRun))]
        else:
            obj1IX = Y.index[(Y['label'] == pair[0])]
            obj2IX = Y.index[(Y['label'] == pair[1])]

        # Get the average of the first object, then the second object
        obj1X = np.mean(X[obj1IX], 0)
        obj2X = np.mean(X[obj2IX], 0)

        # Build the importance map
        mult1X = obj1X * B
        mult2X = obj2X * B

        # Sort these so that they are from least to most important for a given category.
        sortmult1X = mult1X.argsort()[::-1]
        sortmult2X = mult2X.argsort()

        # add to a dictionary for later use
        inds[clf.classes_[0]] = sortmult1X
        inds[clf.classes_[1]] = sortmult2X
                 
        return inds

    if 'milgram' in os.getcwd():
        main_dir='/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/'
    else:
        main_dir='/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtcloud_kp/'

    working_dir=main_dir
    os.chdir(working_dir)

    '''
    if you read runRecording for current session and found that there are only 4 runs in the current session, 
    you read the runRecording for previous session and fetch the last 4 recognition runs from previous session
    '''
    runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])]) # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]
    if len(actualRuns) < 8:
        runRecording_preDay = pd.read_csv(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/../runRecording.csv")
        actualRuns_preDay = list(runRecording_preDay['run'].iloc[list(np.where(1==1*(runRecording_preDay['type']=='recognition'))[0])])[-(8-len(actualRuns)):] # might be [5,6,7,8]
    else: 
        actualRuns_preDay = []

    assert len(actualRuns_preDay)+len(actualRuns)==8 

    objects = ['bed', 'bench', 'chair', 'table']

    for ii,run in enumerate(actualRuns): # load behavior and brain data for current session
        t = np.load(f"{cfg.recognition_dir}brain_run{run}.npy")
        mask = nib.load(f"{cfg.chosenMask}").get_data()
        t = t[:,mask==1]
        brain_data=t if ii==0 else np.concatenate((brain_data,t), axis=0)

        t = pd.read_csv(f"{cfg.recognition_dir}behav_run{run}.csv")
        behav_data=t if ii==0 else pd.concat([behav_data,t])

    for ii,run in enumerate(actualRuns_preDay): # load behavior and brain data for previous session
        t = np.load(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/brain_run{run}.npy")
        mask = nib.load(f"{cfg.chosenMask}").get_data()
        t = t[:,mask==1]
        brain_data = np.concatenate((brain_data,t), axis=0)

        t = pd.read_csv(f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/behav_run{run}.csv")
        behav_data = pd.concat([behav_data,t])

    FEAT=brain_data.reshape(brain_data.shape[0],-1)
    print(f"FEAT.shape={FEAT.shape}")
    FEAT_mean=np.mean(FEAT,axis=1)
    FEAT=(FEAT.T-FEAT_mean).T

    META=behav_data

    # convert item colume to label colume
    imcodeDict={
    'A': 'bed',
    'B': 'chair',
    'C': 'table',
    'D': 'bench'}
    label=[]
    for curr_trial in range(META.shape[0]):
        label.append(imcodeDict[META['Item'].iloc[curr_trial]])
    META['label']=label # merge the label column with the data dataframe

    # Which run to use as test data (leave as None to not have test data)
    testRun = 2 # when testing: testRun = 2 ; META['run_num'].iloc[:5]=2

    # Decide on the proportion of crescent data to use for classification
    include = 1
    accuracyContainer=[]


    allpairs = itertools.combinations(objects,2)

    # Iterate over all the possible target pairs of objects
    for pair in allpairs:
        # Find the control (remaining) objects for this pair
        altpair = other(pair)
        
        # pull sorted indices for each of the critical objects, in order of importance (low to high)
        # inds = get_inds(FEAT, META, pair, testRun=testRun)
        
        # Find the number of voxels that will be left given your inclusion parameter above
        # nvox = red_vox(FEAT.shape[1], include)
        
        for obj in pair:
            # foil = [i for i in pair if i != obj][0]
            for altobj in altpair:
                
                # establish a naming convention where it is $TARGET_$CLASSIFICATION
                # Target is the NF pair (e.g. bed/bench)
                # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
                naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj)
                
                # Pull the relevant inds from your previously established dictionary 
                # obj_inds = inds[obj]
                
                # If you're using testdata, this function will split it up. Otherwise it leaves out run as a parameter
                # if testRun:
                #     trainIX = META.index[(META['label'].isin([obj, altobj])) & (META['run_num'] != int(testRun))]
                #     testIX = META.index[(META['label'].isin([obj, altobj])) & (META['run_num'] == int(testRun))]
                # else:
                #     trainIX = META.index[(META['label'].isin([obj, altobj]))]
                #     testIX = META.index[(META['label'].isin([obj, altobj]))]
                # # pull training and test data
                # trainX = FEAT[trainIX]
                # testX = FEAT[testIX]
                # trainY = META.iloc[trainIX].label
                # testY = META.iloc[testIX].label
                
                # print(f"obj={obj},altobj={altobj}")
                # print(f"unique(trainY)={np.unique(trainY)}")
                # print(f"unique(testY)={np.unique(testY)}")
                # assert len(np.unique(trainY))==2

                if testRun:
                    trainIX = ((META['label']==obj) + (META['label']==altobj)) * (META['run_num']!=int(testRun))
                    testIX = ((META['label']==obj) + (META['label']==altobj)) * (META['run_num']==int(testRun))
                else:
                    trainIX = ((META['label']==obj) + (META['label']==altobj))
                    testIX = ((META['label']==obj) + (META['label']==altobj))
                # pull training and test data
                trainX = FEAT[trainIX]
                testX = FEAT[testIX]
                trainY = META.iloc[np.asarray(trainIX)].label
                testY = META.iloc[np.asarray(testIX)].label

                print(f"obj={obj},altobj={altobj}")
                print(f"unique(trainY)={np.unique(trainY)}")
                print(f"unique(testY)={np.unique(testY)}")
                assert len(np.unique(trainY))==2

                # # If you're selecting high-importance features, this bit handles that
                # if include < 1:
                #     trainX = trainX[:, obj_inds[-nvox:]]
                #     testX = testX[:, obj_inds[-nvox:]]
                
                # Train your classifier
                clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                         multi_class='multinomial').fit(trainX, trainY)
                
                model_folder = cfg.trainingModel_dir
                # Save it for later use
                joblib.dump(clf, model_folder +'/{}.joblib'.format(naming))
                
                # Monitor progress by printing accuracy (only useful if you're running a test set)
                acc = clf.score(testX, testY)
                print(naming, acc)

def behaviorDataLoading(cfg,curr_run):
    '''
    extract the labels which is selected by the subject and coresponding TR and time
    check if the subject's response is correct. When Item is A,bed, response should be 1, or it is wrong
    '''
    behav_data = pd.read_csv(f"{cfg.recognition_dir}{cfg.subjectName}_{curr_run}.csv")

    # the item(imcode) colume of the data represent each image in the following correspondence
    imcodeDict={
    'A': 'bed',
    'B': 'chair',
    'C': 'table',
    'D': 'bench'}

    # When the imcode code is "A", the correct response should be '1', "B" should be '2'
    correctResponseDict={
    'A': 1,
    'B': 2,
    'C': 1,
    'D': 2}

    # extract the labels which is selected by the subject and coresponding TR and time
    behav_data = behav_data[['TR', 'image_on', 'Resp',  'Item']] # the TR, the real time it was presented, 
    behav_data=behav_data.dropna(subset=['Item'])

    # check if the subject's response is correct. When Item is A,bed, response should be 1, or it is wrong
    isCorrect=[]
    for curr_trial in range(behav_data.shape[0]):
        isCorrect.append(correctResponseDict[behav_data['Item'].iloc[curr_trial]]==behav_data['Resp'].iloc[curr_trial])

    behav_data['isCorrect']=isCorrect # merge the isCorrect clumne with the data dataframe
    behav_data['subj']=[cfg.subjectName for i in range(len(behav_data))]
    behav_data['run_num']=[int(curr_run) for i in range(len(behav_data))]
    behav_data=behav_data[behav_data['isCorrect']] # discard the trials where the subject made wrong selection
    return behav_data




def recognition_preprocess_2run(cfg,run_asTemplate): 
    '''
    purpose: 
        prepare the data for 2 recognition runs     (to later(not in this function) get the morphing target function)
        find the functional template image for current session
    steps:
        convert all dicom files into nii files in the temp dir. 
        find the middle volume of the run1 as the template volume, convert this to the previous template volume space and save the converted file as today's functional template (templateFunctionalVolume)
        align every other functional volume with templateFunctionalVolume (3dvolreg)
    '''
    from shutil import copyfile
    from rtCommon.imageHandling import convertDicomFileToNifti
    # convert all dicom files into nii files in the temp dir. 
    tmp_dir=f"{cfg.tmp_folder}{time.time()}/" ; mkdir(tmp_dir)
    dicomFiles=glob(f"{cfg.dicom_dir}/*.dcm") ; dicomFiles.sort()
    for curr_dicom in dicomFiles:
        # dicomImg = readDicomFromFile(curr_dicom) # read dicom file
        dicomFilename=f"{tmp_dir}{curr_dicom.split('/')[-1]}"
        copyfile(curr_dicom,dicomFilename)
        niftiFilename = dicomFilename[:-4]+'.nii'
        convertDicomFileToNifti(dicomFilename, niftiFilename)
        # convertDicomImgToNifti(dicomImg, dicomFilename=f"{tmp_dir}{curr_dicom.split('/')[-1]}") #convert dicom to nii    
        # os.remove(f"{tmp_dir}/{curr_dicom.split('/')[-1]}") # remove temp dcm file

    # find the middle volume of the run1 as the template volume
    # here you are assuming that the first run is a good run
    run_asTemplate=str(run_asTemplate).zfill(6)
    tmp=glob(f"{tmp_dir}001_{run_asTemplate}*.nii") ; tmp.sort()
    # call(f"cp {tmp[int(len(tmp)/2)]} {cfg.recognition_dir}t.nii", shell=True)

    # convert cfg.templateFunctionalVolume to the previous template volume space 
    call(f"flirt -ref {cfg.templateFunctionalVolume} \
        -in {tmp[int(len(tmp)/2)]} \
        -out {cfg.templateFunctionalVolume_converted}",shell=True) 
        
    # align every other functional volume with templateFunctionalVolume (3dvolreg)
    allTRs=glob(f"{tmp_dir}001_*.nii") ; allTRs.sort()

    # select a list of run IDs based on the runRecording.csv, actualRuns would be [1,2] is the 1st and the 3rd runs are recognition runs.
    runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])])[:2]
    for curr_run in actualRuns:
        outputFileNames=[]
        runTRs=glob(f"{tmp_dir}001_{str(curr_run).zfill(6)}_*.nii") ; runTRs.sort()
        for curr_TR in runTRs:
            command = f"3dvolreg \
                -base {cfg.templateFunctionalVolume_converted} \
                -prefix  {curr_TR[0:-4]}_aligned.nii \
                {curr_TR}"
            call(command,shell=True)
            outputFileNames.append(f"{curr_TR[0:-4]}_aligned.nii")
        files=''
        for f in outputFileNames:
            files=files+' '+f
        command=f"fslmerge -t {cfg.recognition_dir}run{curr_run}.nii {files}"
        print('running',command)
        call(command, shell=True)

    # remove the tmp folder
    shutil.rmtree(tmp_dir)
            
    '''
    for each run, 
        load behavior data 
        push the behavior data back for 2 TRs
        save the brain TRs with images
        save the behavior data
    '''

    for curr_run_behav,curr_run in enumerate(actualRuns):
        # load behavior data
        behav_data = behaviorDataLoading(cfg,curr_run_behav+1)

        # brain data is first aligned by pushed back 2TR(4s)
        brain_data = nib.load(f"{cfg.recognition_dir}run{curr_run}.nii.gz").get_data() ; brain_data=np.transpose(brain_data,(3,0,1,2))
        Brain_TR=np.arange(brain_data.shape[0])
        Brain_TR = Brain_TR+2

        # select volumes of brain_data by counting which TR is left in behav_data
        Brain_TR=Brain_TR[list(behav_data['TR'])] # original TR begin with 0
        if Brain_TR[-1]>=brain_data.shape[0]: # when the brain data is not as long as the behavior data, delete the last row
            Brain_TR = Brain_TR[:-1]
            behav_data = behav_data.drop([behav_data.iloc[-1].TR])
        brain_data=brain_data[Brain_TR]
        np.save(f"{cfg.recognition_dir}brain_run{curr_run}.npy", brain_data)
        # save the behavior data
        behav_data.to_csv(f"{cfg.recognition_dir}behav_run{curr_run}.csv")




def morphingTarget(cfg):
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

    import os
    import numpy as np
    import pandas as pd
    import joblib
    import nibabel as nib



    if 'milgram' in os.getcwd():
        main_dir='/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/'
    else:
        main_dir='/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtcloud_kp/'

    working_dir=main_dir
    os.chdir(working_dir)

    runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])]) # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]

    objects = ['bed', 'bench', 'chair', 'table']

    for ii,run in enumerate(actualRuns[:2]): # load behavior and brain data for current session
        t = np.load(f"{cfg.recognition_dir}brain_run{run}.npy")
        mask = nib.load(f"{cfg.chosenMask}").get_data()
        t = t[:,mask==1]
        brain_data=t if ii==0 else np.concatenate((brain_data,t), axis=0)

        t = pd.read_csv(f"{cfg.recognition_dir}behav_run{run}.csv")
        behav_data=t if ii==0 else pd.concat([behav_data,t])

    FEAT=brain_data.reshape(brain_data.shape[0],-1)
    FEAT_mean=np.mean(FEAT,axis=1)
    FEAT=(FEAT.T-FEAT_mean).T
    META=behav_data

    # convert item colume to label colume
    imcodeDict={
    'A': 'bed',
    'B': 'chair',
    'C': 'table',
    'D': 'bench'}
    label=[]
    for curr_trial in range(META.shape[0]):
        label.append(imcodeDict[META['Item'].iloc[curr_trial]])
    META['label']=label # merge the label column with the data dataframe


    def classifierEvidence(clf,X,Y): # X shape is [trials,voxelNumber], Y is ['bed', 'bed'] for example # return a 1-d array of probability
        # This function get the data X and evidence object I want to know Y, and output the trained model evidence.
        targetID=[np.where((clf.classes_==i)==True)[0][0] for i in Y]
        Evidence=(np.sum(X*clf.coef_,axis=1)+clf.intercept_) if targetID[0]==1 else (1-(np.sum(X*clf.coef_,axis=1)+clf.intercept_))
        return np.asarray(Evidence)

    A_ID = (META['label']=='bed')
    X = FEAT[A_ID]

    # evidence_floor is C evidence for AC_CD BC_CD CD_CD classifier(can also be D evidence for CD classifier)
    Y = ['table'] * X.shape[0]
    CD_clf=joblib.load(cfg.usingModel_dir +'bedbench_benchtable.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
    CD_C_evidence = classifierEvidence(CD_clf,X,Y)
    evidence_floor = np.mean(CD_C_evidence)
    print(f"evidence_floor={evidence_floor}")

    # evidence_ceil  is A evidence in AC and AD classifier
    Y = ['bed'] * X.shape[0]
    AC_clf=joblib.load(cfg.usingModel_dir +'bedbench_bedtable.joblib') # These 4 clf are the same:   bedbench_bedtable.joblib bedchair_bedtable.joblib benchtable_tablebed.joblib chairtable_tablebed.joblib
    AC_A_evidence = classifierEvidence(AC_clf,X,Y)
    evidence_ceil1 = AC_A_evidence

    Y = ['bed'] * X.shape[0]
    AD_clf=joblib.load(cfg.usingModel_dir +'bedchair_bedbench.joblib') # These 4 clf are the same:   bedchair_bedbench.joblib bedtable_bedbench.joblib benchchair_benchbed.joblib benchtable_benchbed.joblib
    AD_A_evidence = classifierEvidence(AD_clf,X,Y)
    evidence_ceil2 = AD_A_evidence

    evidence_ceil = np.mean((evidence_ceil1+evidence_ceil2)/2)
    print(f"evidence_ceil={evidence_ceil}")

    return evidence_floor, evidence_ceil

    # allpairs = itertools.combinations(objects,2)

    # # Iterate over all the possible target pairs of objects
    # for pair in allpairs:
    #     # Find the control (remaining) objects for this pair
    #     altpair = other(pair)
       
    #     for obj in pair:
    #         # foil = [i for i in pair if i != obj][0]
    #         for altobj in altpair:
                
    #             # establish a naming convention where it is $TARGET_$CLASSIFICATION
    #             # Target is the NF pair (e.g. bed/bench)
    #             # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
    #             naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj)
              

    #             if testRun:
    #                 trainIX = ((META['label']==obj) + (META['label']==altobj)) * (META['run_num']!=int(testRun))
    #                 testIX = ((META['label']==obj) + (META['label']==altobj)) * (META['run_num']==int(testRun))
    #             else:
    #                 trainIX = ((META['label']==obj) + (META['label']==altobj))
    #                 testIX = ((META['label']==obj) + (META['label']==altobj))
    #             # pull training and test data
    #             trainX = FEAT[trainIX]
    #             testX = FEAT[testIX]
    #             trainY = META.iloc[np.asarray(trainIX)].label
    #             testY = META.iloc[np.asarray(testIX)].label

    #             print(f"obj={obj},altobj={altobj}")
    #             print(f"unique(trainY)={np.unique(trainY)}")
    #             print(f"unique(testY)={np.unique(testY)}")
    #             assert len(np.unique(trainY))==2

    #             # # If you're selecting high-importance features, this bit handles that
    #             # if include < 1:
    #             #     trainX = trainX[:, obj_inds[-nvox:]]
    #             #     testX = testX[:, obj_inds[-nvox:]]
                
    #             # Train your classifier
    #             clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
    #                                      multi_class='multinomial').fit(trainX, trainY)
                
    #             model_folder = cfg.trainingModel_dir
    #             # Save it for later use
    #             joblib.dump(clf, model_folder +'/{}.joblib'.format(naming))
                
    #             # Monitor progress by printing accuracy (only useful if you're running a test set)
    #             acc = clf.score(testX, testY)
    #             print(naming, acc)


def Wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))
        

def fetchXnat(sess_ID):
    "rtSynth_sub001"
    "rtSynth_sub001_ses2"
    import subprocess
    from subprocess import call
    rawPath="/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/"
    proc = subprocess.Popen([f'sbatch {rawPath}../fetchXNAT.sh {sess_ID}'],shell=True)

    Wait(f"{rawPath}{sess_ID}.zip")
    call(f"unzip {rawPath}{sess_ID}.zip")
    time.sleep(10)
    proc = subprocess.Popen([f'sbatch {rawPath}../change2nifti.sh {sess_ID}'],shell=True)

    # furthur work need to be done with this resulting nifti folder