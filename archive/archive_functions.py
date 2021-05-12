
# def morphingTarget(cfg):
#     '''
#     purpose:
#         get the morphing target function
#     steps:
#         load train clf
#         load brain data and behavior data
#         get the morphing target function
#             evidence_floor is C evidence for CD classifier(can also be D evidence for CD classifier)
#             evidence_ceil  is A evidence in AC and AD classifier
#     '''

#     import os
#     import numpy as np
#     import pandas as pd
#     import joblib
#     import nibabel as nib



#     if 'milgram' in os.getcwd():
#         main_dir='/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/'
#     else:
#         main_dir='/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtcloud_kp/'

#     working_dir=main_dir
#     os.chdir(working_dir)

#     runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
#     actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])]) # can be [1,2,3,4,5,6,7,8] or [1,2,4,5]

#     objects = ['bed', 'bench', 'chair', 'table']

#     for ii,run in enumerate(actualRuns[:2]): # load behavior and brain data for current session
#         t = np.load(f"{cfg.recognition_dir}brain_run{run}.npy")
#         # mask = nib.load(f"{cfg.chosenMask}").get_data()
#         mask = np.load(cfg.chosenMask)
#         t = t[:,mask==1]
#         t = normalize(t)
#         brain_data=t if ii==0 else np.concatenate((brain_data,t), axis=0)

#         t = pd.read_csv(f"{cfg.recognition_dir}behav_run{run}.csv")
#         behav_data=t if ii==0 else pd.concat([behav_data,t])

#     # FEAT=brain_data.reshape(brain_data.shape[0],-1)
#     FEAT=brain_data
#     print("FEAT.shape=",FEAT.shape)
#     assert len(FEAT.shape)==2
#     # FEAT_mean=np.mean(FEAT,axis=1)
#     # FEAT=(FEAT.T-FEAT_mean).T
#     # FEAT_mean=np.mean(FEAT,axis=0)
#     # FEAT=FEAT-FEAT_mean

#     META=behav_data

#     # convert item colume to label colume
#     imcodeDict={
#     'A': 'bed',
#     'B': 'chair',
#     'C': 'table',
#     'D': 'bench'}
#     label=[]
#     for curr_trial in range(META.shape[0]):
#         label.append(imcodeDict[META['Item'].iloc[curr_trial]])
#     META['label']=label # merge the label column with the data dataframe




#     def clf_score(obj,altobj,clf,FEAT,META): # obj is A, altobj is B, clf is AC_clf
#         ID = (META['label']==imcodeDict[obj]) | (META['label']==imcodeDict[altobj])
#         X = FEAT[ID]
#         Y = META['label'][ID]
#         acc = clf.score(X, Y)
#         print(f"{obj}{altobj}_clf accuracy = {acc}")

#     A_ID = (META['label']=='bed')
#     X = FEAT[A_ID]

#     # evidence_floor is C evidence for AC_CD BC_CD CD_CD classifier(can also be D evidence for CD classifier)




#     #try out other forms of floor: C evidence in AC and D evidence for AD

#     # imcodeDict={
#     # 'A': 'bed',
#     # 'B': 'chair',
#     # 'C': 'table',
#     # 'D': 'bench'}

#     # this part is to know the performance of BC and BD clf on current day to judge whether to use both clf in realtime.
#     print("BC_clf BD_clf accuracy")

#     BC_clf=joblib.load(cfg.usingModel_dir +'bedchair_chairtable.joblib') # These 4 clf are the same:   bedchair_bedbench.joblib bedtable_bedbench.joblib benchchair_benchbed.joblib benchtable_benchbed.joblib
#     clf_score("B","C",BC_clf,FEAT,META)
#     B_ID = (META['label']=='chair')
#     BC_B_evidence = np.mean(classifierEvidence(BC_clf,FEAT[B_ID],'chair'))
#     print(f"B evidence for BC_clf when B is presented={BC_B_evidence}")

#     BD_clf=joblib.load(cfg.usingModel_dir +'bedchair_chairbench.joblib') # These 4 clf are the same:   bedchair_bedbench.joblib bedtable_bedbench.joblib benchchair_benchbed.joblib benchtable_benchbed.joblib
#     clf_score("B","D",BD_clf,FEAT,META)
#     B_ID = (META['label']=='chair')
#     BD_B_evidence = np.mean(classifierEvidence(BD_clf,FEAT[B_ID],'chair'))
#     print(f"B evidence for BD_clf when B is presented={BD_B_evidence}")

#     print()

#     print("floor")
#     # D evidence for AD_clf when A is presented.
#     Y = 'bench'
#     AD_clf=joblib.load(cfg.usingModel_dir +'bedchair_bedbench.joblib') # These 4 clf are the same:   bedchair_bedbench.joblib bedtable_bedbench.joblib benchchair_benchbed.joblib benchtable_benchbed.joblib
#     clf_score("A","D",AD_clf,FEAT,META)
#     AD_D_evidence = classifierEvidence(AD_clf,X,Y)
#     evidence_floor = np.mean(AD_D_evidence)
#     print(f"D evidence for AD_clf when A is presented={evidence_floor}")

#     # C evidence for AC_clf when A is presented.
#     Y = 'table'
#     AC_clf=joblib.load(cfg.usingModel_dir +'benchtable_tablebed.joblib') # These 4 clf are the same:   bedbench_bedtable.joblib bedchair_bedtable.joblib benchtable_tablebed.joblib chairtable_tablebed.joblib
#     clf_score("A","C",AC_clf,FEAT,META)
#     AC_C_evidence = classifierEvidence(AC_clf,X,Y)
#     evidence_floor = np.mean(AC_C_evidence)
#     print(f"C evidence for AC_clf when A is presented={evidence_floor}")


#     # D evidence for CD_clf when A is presented.
#     Y = 'bench'
#     CD_clf=joblib.load(cfg.usingModel_dir +'bedbench_benchtable.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
#     clf_score("C","D",CD_clf,FEAT,META)
#     CD_D_evidence = classifierEvidence(CD_clf,X,Y)
#     evidence_floor = np.mean(CD_D_evidence)
#     print(f"D evidence for CD_clf when A is presented={evidence_floor}")

#     # C evidence for CD_clf when A is presented.
#     Y = 'table'
#     CD_clf=joblib.load(cfg.usingModel_dir +'bedbench_benchtable.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
#     clf_score("C","D",CD_clf,FEAT,META)
#     CD_C_evidence = classifierEvidence(CD_clf,X,Y)
#     evidence_floor = np.mean(CD_C_evidence)
#     print(f"C evidence for CD_clf when A is presented={evidence_floor}")
#     evidence_floor = 0




#     print("ceil")
#     # evidence_ceil  is A evidence in AC and AD classifier
#     Y = 'bed'
#     AC_clf=joblib.load(cfg.usingModel_dir +'benchtable_tablebed.joblib') # These 4 clf are the same:   bedbench_bedtable.joblib bedchair_bedtable.joblib benchtable_tablebed.joblib chairtable_tablebed.joblib
#     clf_score("A","C",AC_clf,FEAT,META)
#     AC_A_evidence = classifierEvidence(AC_clf,X,Y)
#     evidence_ceil1 = AC_A_evidence
#     print(f"A evidence in AC_clf when A is presented={np.mean(evidence_ceil1)}")

#     Y = 'bed'
#     AD_clf=joblib.load(cfg.usingModel_dir +'bedchair_bedbench.joblib') # These 4 clf are the same:   bedchair_bedbench.joblib bedtable_bedbench.joblib benchchair_benchbed.joblib benchtable_benchbed.joblib
#     clf_score("A","D",AD_clf,FEAT,META)
#     AD_A_evidence = classifierEvidence(AD_clf,X,Y)
#     evidence_ceil2 = AD_A_evidence
#     print(f"A evidence in AD_clf when A is presented={np.mean(evidence_ceil2)}")

#     # evidence_ceil = np.mean(evidence_ceil1)
#     # evidence_ceil = np.mean(evidence_ceil2)
#     evidence_ceil = np.mean((evidence_ceil1+evidence_ceil2)/2)
#     print(f"evidence_ceil={evidence_ceil}")

#     return evidence_floor, evidence_ceil

#     # allpairs = itertools.combinations(objects,2)

#     # # Iterate over all the possible target pairs of objects
#     # for pair in allpairs:
#     #     # Find the control (remaining) objects for this pair
#     #     altpair = other(pair)
       
#     #     for obj in pair:
#     #         # foil = [i for i in pair if i != obj][0]
#     #         for altobj in altpair:
                
#     #             # establish a naming convention where it is $TARGET_$CLASSIFICATION
#     #             # Target is the NF pair (e.g. bed/bench)
#     #             # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
#     #             naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj)
              

#     #             if testRun:
#     #                 trainIX = ((META['label']==obj) + (META['label']==altobj)) * (META['run_num']!=int(testRun))
#     #                 testIX = ((META['label']==obj) + (META['label']==altobj)) * (META['run_num']==int(testRun))
#     #             else:
#     #                 trainIX = ((META['label']==obj) + (META['label']==altobj))
#     #                 testIX = ((META['label']==obj) + (META['label']==altobj))
#     #             # pull training and test data
#     #             trainX = FEAT[trainIX]
#     #             testX = FEAT[testIX]
#     #             trainY = META.iloc[np.asarray(trainIX)].label
#     #             testY = META.iloc[np.asarray(testIX)].label

#     #             print(f"obj={obj},altobj={altobj}")
#     #             print(f"unique(trainY)={np.unique(trainY)}")
#     #             print(f"unique(testY)={np.unique(testY)}")
#     #             assert len(np.unique(trainY))==2

#     #             # # If you're selecting high-importance features, this bit handles that
#     #             # if include < 1:
#     #             #     trainX = trainX[:, obj_inds[-nvox:]]
#     #             #     testX = testX[:, obj_inds[-nvox:]]
                
#     #             # Train your classifier
#     #             clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
#     #                                      multi_class='multinomial').fit(trainX, trainY)
                
#     #             model_folder = cfg.trainingModel_dir
#     #             # Save it for later use
#     #             joblib.dump(clf, model_folder +'/{}.joblib'.format(naming))
                
#     #             # Monitor progress by printing accuracy (only useful if you're running a test set)
#     #             acc = clf.score(testX, testY)
#     #             print(naming, acc)


# def classifierEvidence(clf,X,Y): # X shape is [trials,voxelNumber], Y is ['bed', 'bed'] for example # return a 1-d array of probability
#     # This function get the data X and evidence object I want to know Y, and output the trained model evidence.
#     targetID=[np.where((clf.classes_==i)==True)[0][0] for i in Y]
#     # Evidence=(np.sum(X*clf.coef_,axis=1)+clf.intercept_) if targetID[0]==1 else (1-(np.sum(X*clf.coef_,axis=1)+clf.intercept_))
#     Evidence=(X@clf.coef_.T+clf.intercept_) if targetID[0]==1 else (-(X@clf.coef_.T+clf.intercept_))
#     Evidence = 1/(1+np.exp(-Evidence))
#     return np.asarray(Evidence)

# def classifierEvidence(clf,X,Y):
#     ID=np.where((clf.classes_==Y)*1==1)[0][0]
#     p = clf.predict_proba(X)[:,ID]
#     BX=np.log(p/(1-p))
#     return BX

# def classifierEvidence(clf,X,Y):
#     ID=np.where((clf.classes_==Y)*1==1)[0][0]
#     Evidence=(X@clf.coef_.T+clf.intercept_) if ID==1 else (-(X@clf.coef_.T+clf.intercept_))
#     # Evidence=(X@clf.coef_.T+clf.intercept_) if ID==0 else (-(X@clf.coef_.T+clf.intercept_))
#     return np.asarray(Evidence)


    # def gaussian(x, mu, sig):
    #     # mu and sig is determined before each neurofeedback session using 2 recognition runs.
    #     return round(1+18*(1 - np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))) # map from (0,1) -> [1,19]

    # def jitter(size,const=0):
    #     jit = np.random.normal(0+const, 0.05, size)
    #     X = np.zeros((size))
    #     X = X + jit
    #     return X
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

    # def evidence(trainX,trainY):
    #     # def classifierEvidence(clf,X,Y):
    #     #     ID=np.where((clf.classes_==Y[0])*1==1)[0][0]
    #     #     Evidence=(X@clf.coef_.T+clf.intercept_) if ID==1 else (-(X@clf.coef_.T+clf.intercept_))
    #     #     # Evidence=(X@clf.coef_.T+clf.intercept_) if ID==0 else (-(X@clf.coef_.T+clf.intercept_))
    #     #     return np.asarray(Evidence)
    #     FEAT=trainX
    #     META=trainY
        
    #     A_ID = META['label']=='bed'
    #     X = FEAT[A_ID]

    #     print("floor")
    #     # D evidence for AD_clf when A is presented.
    #     Y = 'bench'
    #     AD_clf=joblib.load(cfg.trainingModel_dir +'bedchair_bedbench.joblib') # These 4 clf are the same:   bedchair_bedbench.joblib bedtable_bedbench.joblib benchchair_benchbed.joblib benchtable_benchbed.joblib
    #     AD_D_evidence = classifierEvidence(AD_clf,X,Y)
    #     evidence_floor = np.mean(AD_D_evidence)
    #     print(f"D evidence for AD_clf when A is presented={evidence_floor}")

    #     # C evidence for AC_clf when A is presented.
    #     Y = 'table'
    #     AC_clf=joblib.load(cfg.trainingModel_dir +'benchtable_tablebed.joblib') # These 4 clf are the same:   bedbench_bedtable.joblib bedchair_bedtable.joblib benchtable_tablebed.joblib chairtable_tablebed.joblib
    #     AC_C_evidence = classifierEvidence(AC_clf,X,Y)
    #     evidence_floor = np.mean(AC_C_evidence)
    #     print(f"C evidence for AC_clf when A is presented={evidence_floor}")


    #     # D evidence for CD_clf when A is presented.
    #     Y = 'bench'
    #     CD_clf=joblib.load(cfg.trainingModel_dir +'bedbench_benchtable.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
    #     CD_D_evidence = classifierEvidence(CD_clf,X,Y)
    #     evidence_floor = np.mean(CD_D_evidence)
    #     print(f"D evidence for CD_clf when A is presented={evidence_floor}")

    #     # C evidence for CD_clf when A is presented.
    #     Y = 'table'
    #     CD_clf=joblib.load(cfg.trainingModel_dir +'bedbench_benchtable.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
    #     CD_C_evidence = classifierEvidence(CD_clf,X,Y)
    #     evidence_floor = np.mean(CD_C_evidence)
    #     print(f"C evidence for CD_clf when A is presented={evidence_floor}")

    #     # since this subject has CD_clf C evidence systematically too high(sometimes higher than AC_clf A evidence or AD_clf A evidence), I choose to use 0 as the floor
    #     evidence_floor = 0




    #     print("ceil")
    #     # evidence_ceil  is A evidence in AC and AD classifier
    #     Y = 'bed'
    #     AC_clf=joblib.load(cfg.trainingModel_dir +'benchtable_tablebed.joblib') # These 4 clf are the same:   bedbench_bedtable.joblib bedchair_bedtable.joblib benchtable_tablebed.joblib chairtable_tablebed.joblib
    #     AC_A_evidence = classifierEvidence(AC_clf,X,Y)
    #     evidence_ceil1 = AC_A_evidence
    #     print(f"A evidence in AC_clf when A is presented={np.mean(evidence_ceil1)}")

    #     Y = 'bed'
    #     AD_clf=joblib.load(cfg.trainingModel_dir +'bedchair_bedbench.joblib') # These 4 clf are the same:   bedchair_bedbench.joblib bedtable_bedbench.joblib benchchair_benchbed.joblib benchtable_benchbed.joblib
    #     AD_A_evidence = classifierEvidence(AD_clf,X,Y)
    #     evidence_ceil2 = AD_A_evidence
    #     print(f"A evidence in AD_clf when A is presented={np.mean(evidence_ceil2)}")

    #     # evidence_ceil = np.mean(evidence_ceil1)
    #     # evidence_ceil = np.mean(evidence_ceil2)
    #     evidence_ceil = np.mean((evidence_ceil1+evidence_ceil2)/2)
    #     print(f"evidence_ceil={evidence_ceil}")

    #     mu = (evidence_ceil+evidence_floor)/2
    #     sig = (evidence_ceil-evidence_floor)/2.3548
    #     print(f"floor={evidence_floor}, ceil={evidence_ceil}")
    #     print(f"mu={mu}, sig={sig}")


    # # print the evidence using model training data
    # evidence(FEAT,META)
    # print the evidence using model testing data
    



def recognition_preprocess_2run(cfg,scan_asTemplate): 
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
    # 把cfg.dicom_dir的file复制到tmp folder并且转换成nii
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
    scan_asTemplate=str(scan_asTemplate).zfill(6)
    tmp=glob(f"{tmp_dir}001_{scan_asTemplate}*.nii") ; tmp.sort()
    # print(f"all nii files: {tmp}")
    # call(f"cp {tmp[int(len(tmp)/2)]} {cfg.recognition_dir}t.nii", shell=True)

    # convert cfg.templateFunctionalVolume to the previous template volume space 
    cmd=f"flirt -ref {cfg.templateFunctionalVolume} \
        -in {tmp[int(len(tmp)/2)]} \
        -out {cfg.templateFunctionalVolume_converted}"
    print(cmd)
    call(cmd,shell=True) 
        
    # align every other functional volume with templateFunctionalVolume (3dvolreg)
    allTRs=glob(f"{tmp_dir}001_*.nii") ; allTRs.sort()

    # select a list of run IDs based on the runRecording.csv, actualRuns would be [1,2] is the 1st and the 3rd runs are recognition runs.
    runRecording = pd.read_csv(f"{cfg.recognition_dir}../runRecording.csv")
    actualRuns = list(runRecording['run'].iloc[list(np.where(1==1*(runRecording['type']=='recognition'))[0])])[:2]
    for curr_run in actualRuns:
        if not (os.path.exists(f"{cfg.recognition_dir}run{curr_run}.nii.gz") and os.path.exists(f"{cfg.recognition_dir}run{curr_run}.nii")):
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

