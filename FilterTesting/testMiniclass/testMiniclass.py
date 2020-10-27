# (in the case of target axis AB) 
# Calculate only the A evidence for A (input A for classifier AC and AD) compared 
# with A evidence for B (input B for classifier AC and AD) ; 
# B evidence for B (input B for classifier BC and BD) compared with B 
# evidence for A (input A for classifier BC and BD)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import joblib
import nibabel as nib
import itertools
from sklearn.linear_model import LogisticRegression
from IPython.display import clear_output
import sys
from subprocess import call
import pickle
import pdb
import time
from tqdm import tqdm

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

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

def classifierEvidence(clf,X,Y): # X shape is [trials,voxelNumber], Y is ['bed', 'bed'] for example # return a list a probability
    # This function get the data X and evidence object I want to know Y, and output the trained model evidence.
    targetID=[np.where((clf.classes_==i)==True)[0][0] for i in Y]
    Evidence=[clf.predict_proba(X[i].reshape(1,-1))[0][j] for i,j in enumerate(targetID)]
    # print('targetID=', targetID)
    # print('Evidence=',Evidence)
    return np.asarray(Evidence)

def _and_(L):
    if len(L)==2:
        return np.logical_and(L[0],L[1])
    else:
        return np.logical_and(L[0],_and_(L[1:]))

def saveNpInDf(array):
    dataDir='./saveNpInDf/'
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)
    fileName=dataDir+str(time.time())
    np.save(fileName,array)
    return fileName

def resample(L):
    L=np.asarray(L)
    sample_mean=[]
    for iter in tqdm(range(10000)):
        resampleID=np.random.choice(len(L), len(L), replace=True)
        resample_acc=L[resampleID]
        sample_mean.append(np.nanmean(resample_acc))
    sample_mean=np.asarray(sample_mean)
    m = np.nanmean(sample_mean,axis=0)
    upper=np.percentile(sample_mean, 97.5, axis=0)
    lower=np.percentile(sample_mean, 2.5, axis=0)
    return m,m-lower,upper-m

def bar(LL,labels=None,title=None):
    import matplotlib.pyplot as plt
    D=np.asarray([resample(L) for L in LL])
    m=D[:,0]
    lower=D[:,1]
    upper=D[:,2]
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10,10))
    ax.bar(x_pos, m, yerr=[lower,upper], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('object evidence')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=30,ha='right')
    plt.show()
    return m,lower,upper
    
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

def getEvidence(sub,testEvidence,METADICT=None,FEATDICT=None,filterType=None,roi="V1",include=1,testRun=6,accuracyContainer=None):
    # each testRun, each subject, each target axis, each target obj would generate one.
    pdb.set_trace()
    META = METADICT[sub]
    print('META.shape=',META.shape)
    FEAT = FEATDICT[sub]
    # Using the trained model, get the evidence
    
    objects=['bed', 'bench', 'chair', 'table']

    allpairs = itertools.combinations(objects,2)
    for pair in allpairs: #pair=('bed', 'bench')
        # Find the control (remaining) objects for this pair
        altpair = other(pair) #altpair=('chair', 'table')
        for obj in pair: #obj='bed'
            # in the current target axis pair=('bed', 'bench') altpair=('chair', 'table'), display image obj='bed'
            # find the evidence for bed from the (bed chair) and (bed table) classifier

            # get the test data and seperate the test data into category obj and category other
            otherObj=[i for i in pair if i!=obj][0]
            print('otherObj=',otherObj)
            objID = META.index[(META['label'].isin([obj])) & (META['run_num'] == int(testRun))]
            otherObjID = META.index[(META['label'].isin([otherObj])) & (META['run_num'] == int(testRun))]
            
            obj_X=FEAT[objID]
            # obj_Y=META.iloc[objID].label
            otherObj_X=FEAT[otherObjID]
            # otherObj_Y=META.iloc[otherObjID].label

            model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{include}/{roi}/{filterType}/{testRun}/'
            print(f'loading {model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[0]}.joblib')
            print(f'loading {model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[1]}.joblib')
            clf1 = joblib.load(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[0]}.joblib')
            clf2 = joblib.load(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[1]}.joblib')

            s1_obj_evidence = classifierEvidence(clf1,obj_X,[obj] * obj_X.shape[0])
            s2_obj_evidence = classifierEvidence(clf2,obj_X,[obj] * obj_X.shape[0])
            obj_evidence = np.mean([s1_obj_evidence, s2_obj_evidence],axis=0) # when axis=0, the output would be obj_evidence for each trial in the testing data.
            print('obj_evidence=',obj_evidence)

            s1_otherObj_evidence = classifierEvidence(clf1,otherObj_X,[obj] * otherObj_X.shape[0])
            s2_otherObj_evidence = classifierEvidence(clf2,otherObj_X,[obj] * otherObj_X.shape[0])
            otherObj_evidence = np.mean([s1_otherObj_evidence, s2_otherObj_evidence],axis=0)
            print('otherObj_evidence=',otherObj_evidence)

            
            # test here
            AC_clf = joblib.load(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[0]}.joblib') #AC classifier
            AD_clf = joblib.load(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[1]}.joblib') #AD classifier
            BC_clf = joblib.load(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{otherObj}{altpair[0]}.joblib') #BC classifier
            BD_clf = joblib.load(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{otherObj}{altpair[1]}.joblib') #BD classifier

            AC_A=classifierEvidence(AC_clf,obj_X,[obj] * obj_X.shape[0]) #AC evidence for A : AC classifier, A evidence
            AD_A=classifierEvidence(AD_clf,obj_X,[obj] * obj_X.shape[0]) #AD evidence for A : AD classifier, A evidence
            BC_A=classifierEvidence(BC_clf,obj_X,[otherObj] * obj_X.shape[0]) #BC evidence for A : BC classifier, B evidence
            BD_A=classifierEvidence(BD_clf,obj_X,[otherObj] * obj_X.shape[0]) #BD evidence for A : BD classifier, B evidence
            a=np.asarray([AC_A,AD_A,BC_A,BD_A]) # 4 x numTrials

            plt.figure()
            bar([a[i,:] for i in range(a.shape[0])],labels=['AC_A','AD_A',"BC_A","BD_A"])

            AC_B=classifierEvidence(AC_clf,otherObj_X,[obj] * obj_X.shape[0]) #AC evidence for B
            AD_B=classifierEvidence(AD_clf,otherObj_X,[obj] * obj_X.shape[0]) #AD evidence for B
            BC_B=classifierEvidence(BC_clf,otherObj_X,[otherObj] * obj_X.shape[0]) #BC evidence for B
            BD_B=classifierEvidence(BD_clf,otherObj_X,[otherObj] * obj_X.shape[0]) #BD evidence for B
            b=np.asarray([AC_B,AD_B,BC_B,BD_B]) # 4 x numTrials
            
            plt.figure()
            bar([b[i,:] for i in range(b.shape[0])],labels=['AC_B','AD_B',"BC_B","BD_B"])

            t=np.concatenate((a, b), axis=1)
            fig, ax = plt.subplots(figsize=(20,5))
            _=ax.plot(t[0,:],'g.')
            _=ax.plot(t[1,:],'g.')
            _=ax.plot(t[2,:],'r+')
            _=ax.plot(t[3,:],'r+')

            # A=obj
            # B=otherObj
            # C=altpair[0]
            # D=altpair[1]

            AC_acc = accuracyContainer[_and_([
                accuracyContainer['targetAxis']==pair, #AB
                accuracyContainer['obj']==obj, #A
                accuracyContainer['altobj']==altpair[0], #C
                accuracyContainer['sub']==sub,
                accuracyContainer['testRun']==testRun,
                accuracyContainer['filterType']==filterType,
                accuracyContainer['include']==include,
                accuracyContainer['roi']==roi
                ])]['acc'].iloc[0]
            AD_acc = accuracyContainer[_and_([
                accuracyContainer['targetAxis']==pair, #AB
                accuracyContainer['obj']==obj, #A
                accuracyContainer['altobj']==altpair[1], #D

                accuracyContainer['sub']==sub,
                accuracyContainer['testRun']==testRun,
                accuracyContainer['filterType']==filterType,
                accuracyContainer['include']==include,
                accuracyContainer['roi']==roi
                ])]['acc'].iloc[0]
            BC_acc = accuracyContainer[_and_([
                accuracyContainer['targetAxis']==pair, #AB
                accuracyContainer['obj']==otherObj, #B
                accuracyContainer['altobj']==altpair[0], #C

                accuracyContainer['sub']==sub,
                accuracyContainer['testRun']==testRun,
                accuracyContainer['filterType']==filterType,
                accuracyContainer['include']==include,
                accuracyContainer['roi']==roi
                ])]['acc'].iloc[0]
            BD_acc = accuracyContainer[_and_([
                accuracyContainer['targetAxis']==pair, #AB
                accuracyContainer['obj']==otherObj, #B
                accuracyContainer['altobj']==altpair[1], #D

                accuracyContainer['sub']==sub,
                accuracyContainer['testRun']==testRun,
                accuracyContainer['filterType']==filterType,
                accuracyContainer['include']==include,
                accuracyContainer['roi']==roi
                ])]['acc'].iloc[0]


            title=f'AC_acc={AC_acc};AD_acc={AD_acc};BC_acc={BC_acc};BD_acc={BD_acc}'
            print(title)
            # plt.title(title)
            ax.set_title(title)
            plt.show()

            

            # accuracyContainer[_and_([
            #     accuracyContainer['sub']==110171,
            #     accuracyContainer['testRun']==6,
            #     accuracyContainer['targetAxis']=="('bed', 'bench')", #AB
            #     accuracyContainer['obj']=='bed', #A
            #     accuracyContainer['altobj']=='chair', #B
            #     accuracyContainer['filterType']=='noFilter',
            #     accuracyContainer['include']==1.,
            #     accuracyContainer['roi']=='V1'
            #     ])]

            # testEvidence = testEvidence.append({
            #     'sub':sub,
            #     'testRun':testRun,
            #     'targetAxis':pair,
            #     'obj':obj,
            #     'obj_evidence':np.mean(obj_evidence),
            #     'otherObj_evidence':np.mean(otherObj_evidence),
            #     'objMinusOther_evidence':np.mean(obj_evidence) - np.mean(otherObj_evidence),
            #     'filterType':filterType,
            #     'include':include,
            #     'roi':roi
            #     },
            #     ignore_index=True)
            testEvidence = testEvidence.append({
                'sub':sub,
                'testRun':testRun,
                'targetAxis':pair,
                'obj':obj,
                
                'AC_A_evidence':saveNpInDf(s1_obj_evidence), # AC evidence for A
                'AD_A_evidence':saveNpInDf(s2_obj_evidence), # AD evidence for A
                'AC_B_evidence':saveNpInDf(s1_otherObj_evidence), # AC evidence for B
                'AD_B_evidence':saveNpInDf(s2_otherObj_evidence), # AD evidence for B

                'obj_evidence':saveNpInDf(obj_evidence),
                'otherObj_evidence':saveNpInDf(otherObj_evidence),
                'objMinusOther_evidence':np.mean(obj_evidence) - np.mean(otherObj_evidence),

                'filterType':filterType,
                'include':include,
                'roi':roi
                },
                ignore_index=True)

    return testEvidence

def minimalClass(filterType = 'noFilter',testRun = 6, roi="V1",include = 1): #include is the proportion of features selected
    
    accuracyContainer = pd.DataFrame(columns=['sub','testRun','targetAxis','obj','altobj','acc','filterType','roi'])
    testEvidence = pd.DataFrame(columns=['sub','testRun','targetAxis','obj','obj_evidence','otherObj_evidence','filterType','roi'])

    # working_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/FilterTesting/neurosketch_realtime_preprocess/'
    # os.chdir(working_dir)

    data_dir=f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/features/{filterType}/recognition/'
    files = os.listdir(data_dir)
    feats = [i for i in files if 'metadata' not in i]
    subjects = np.unique([i.split('_')[0] for i in feats if i.split('_')[0] not in ['1121161','0112174']]) # 1121161 has a grid spacing issue and 0112174 lacks one of regressor file
    # If you want to reduce the number of subjects used for testing purposes
    # subs=len(subjects) # 
    subs=np.arange(1,3)
    subjects = subjects[subs]
    print('subjects=',subjects)

    objects = ['bed', 'bench', 'chair', 'table']
    phases = ['12', '34', '56']

    # THIS CELL READS IN ALL OF THE PARTICIPANTS' DATA and fills into dictionary
    FEATDICT = {}
    METADICT = {}
    subjects_new=[]
    for si, sub in enumerate(subjects[:]):
        try:
            print('{}/{}'.format(si+1, len(subjects)))
            for phase in phases:
                _feat = np.load(data_dir+'/{}_{}_{}_featurematrix.npy'.format(sub, roi, phase))
                _feat = normalize(_feat)
                _meta = pd.read_csv(data_dir+'/metadata_{}_{}_{}.csv'.format(sub, roi, phase))
                if phase!='12':
                    assert _feat.shape[1]==FEAT.shape[1], 'feat shape not matched'
                FEAT = _feat if phase == "12" else np.vstack((FEAT, _feat))
                META = _meta if phase == "12" else pd.concat((META, _meta))
            META = META.reset_index(drop=True)

            assert FEAT.shape[0] == META.shape[0]

            METADICT[sub] = META
            FEATDICT[sub] = FEAT
            clear_output(wait=True)
            subjects_new.append(sub)
        except:
            pass


    # Which run to use as test data (leave as None to not have test data)
    subjects=subjects_new

    # train the models; Decide on the proportion of crescent data to use for classification
    for si,sub in enumerate(subjects):
        # try:
        print('{}/{}'.format(si+1, len(subjects)))
        print(sub)
        META = METADICT[sub]
        FEAT = FEATDICT[sub]
        allpairs = itertools.combinations(objects,2)
        # Iterate over all the possible target pairs of objects
        for pair in allpairs:
            # Find the control (remaining) objects for this pair
            altpair = other(pair)
            
            # pull sorted indices for each of the critical objects, in order of importance (low to high)
            inds = get_inds(FEAT, META, pair, testRun=testRun)
            
            # Find the number of voxels that will be left given your inclusion parameter above
            nvox = red_vox(FEAT.shape[1], include)

            for obj in pair:
                # foil = [i for i in pair if i != obj][0]
                for altobj in altpair:
                    # establish a naming convention where it is $TARGET_$CLASSIFICATION
                    # Target is the NF pair (e.g. bed/bench)
                    # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
                    naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj)
                    
                    # Pull the relevant inds from your previously established dictionary 
                    obj_inds = inds[obj]
                    
                    # If you're using testdata, this function will split it up. Otherwise it leaves out run as a parameter
                    if testRun:
                        trainIX = META.index[(META['label'].isin([obj, altobj])) & (META['run_num'] != int(testRun))]
                        testIX = META.index[(META['label'].isin([obj, altobj])) & (META['run_num'] == int(testRun))]
                    else:
                        trainIX = META.index[(META['label'].isin([obj, altobj]))]
                        testIX = META.index[(META['label'].isin([obj, altobj]))]

                    # pull training and test data
                    trainX = FEAT[trainIX]
                    testX = FEAT[testIX]
                    trainY = META.iloc[trainIX].label
                    testY = META.iloc[testIX].label
                    
                    # If you're selecting high-importance features, this bit handles that
                    if include < 1:
                        trainX = trainX[:, obj_inds[-nvox:]]
                        testX = testX[:, obj_inds[-nvox:]]
                    
                    # Train your classifier
                    # clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                            # multi_class='multinomial').fit(trainX, trainY)
                    # joblib.dump(clf, model_folder + '{}_{}.joblib'.format(sub, naming))

                    clf = joblib.load(model_folder + '{}_{}.joblib'.format(sub, naming)) #AD classifier

                    # clf=model_folder + '{}_{}.joblib'.format(sub, naming)
                    # save_obj(obj_inds[-nvox:],f'{model_folder}{sub}_{naming}.selectedFeatures')

                    # Monitor progress by printing accuracy (only useful if you're running a test set)
                    acc = clf.score(testX, testY)
                    if (si+1)%10==0:
                        print(naming, acc)
                    accuracyContainer = accuracyContainer.append({
                        'sub':sub,
                        'testRun':testRun,
                        'targetAxis':pair, #AB
                        'obj':obj, #A
                        'altobj':altobj,#B
                        'acc':acc,
                        'filterType':filterType,
                        'include':include,
                        'roi':roi
                        },
                        ignore_index=True)

        # except:
        #     pass
    
    for sub in subjects:
        # try:
        testEvidence=getEvidence(sub,testEvidence,
        METADICT=METADICT,
        FEATDICT=FEATDICT,
        filterType=filterType,
        roi=roi,
        include=include,
        testRun=testRun,
        accuracyContainer=accuracyContainer
        )
        # except:
        #     pass
    print('accuracyContainer=',accuracyContainer)
    print('testEvidence=',testEvidence)
    accuracyContainer.to_csv(f"{model_folder}accuracy.csv")
    testEvidence.to_csv(f'{model_folder}testEvidence_.csv')

include=1
roi='V1'
filterType='noFilter'
testRun=6
model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{include}/{roi}/{filterType}/{testRun}/'
print('model_folder=',model_folder)
call(f"mkdir -p {model_folder}",shell=True)
minimalClass(include = include, roi=roi, filterType = filterType, testRun = testRun)


## - to run in parallel
# bash to submit jobs, in the folder of testMiniclass

# # testMiniclass_child.sh
# #!/bin/bash
# #SBATCH --partition=short,scavenge,interactive,long,verylong
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --time=3:00:00
# #SBATCH --job-name=miniClass
# #SBATCH --output=logs/miniClass-%j.out
# #SBATCH --mem-per-cpu=10G
# #SBATCH --mail-type=FAIL
# mkdir -p logs/
# module load miniconda
# source activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtAtten
# include=$1
# roi=$2
# filterType=$3
# testRun=$4
# /usr/bin/time python -u /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/FilterTesting/testMiniclass/testMiniclass.py $include $roi $filterType $testRun


# # testMiniclass_parent.py
# from glob import glob
# import os
# from subprocess import call
# for include in [0.1,0.3,0.6,0.9,1]:
#     for roi in ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']:
#         for filterType in ['noFilter','highPassRealTime','highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']:
#             for testRun in [1,2,3,4,5,6]:
#                 command=f'sbatch testMiniclass_child.sh {include} {roi} {filterType} {testRun}'
#                 print(command)
#             # call(command, shell=True)        


## - load and plot data
def loadPlot():
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    def loadNpInDf(fileName):
        main_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/FilterTesting/neurosketch_realtime_preprocess/'
        return np.load(main_dir+fileName+'.npy')

    def _and_(L):
        if len(L)==2:
            return np.logical_and(L[0],L[1])
        else:
            return np.logical_and(L[0],_and_(L[1:]))

    accuracyContainer=[]
    testEvidence=[]
    for include in [0.1,0.3,0.6,0.9,1]:
        for roi in ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']:
            for filterType in ['noFilter','highPassRealTime','highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']:
                for testRun in [1,2,3,4,5,6]:
                    model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{np.float(include)}/{roi}/{filterType}/{testRun}/'
                    accuracyContainer.append(pd.read_csv(f"{model_folder}accuracy.csv"))
                    testEvidence.append(pd.read_csv(f'{model_folder}testEvidence_.csv'))
    accuracyContainer=pd.concat(accuracyContainer, ignore_index=True)
    testEvidence=pd.concat(testEvidence, ignore_index=True)
    for i in range(len(testEvidence)):
        testEvidence['AC_A_evidence'].iloc[i]=loadNpInDf(testEvidence['AC_A_evidence'].iloc[i])
        testEvidence['AD_A_evidence'].iloc[i]=loadNpInDf(testEvidence['AD_A_evidence'].iloc[i])
        testEvidence['AC_B_evidence'].iloc[i]=loadNpInDf(testEvidence['AC_B_evidence'].iloc[i])
        testEvidence['AD_B_evidence'].iloc[i]=loadNpInDf(testEvidence['AD_B_evidence'].iloc[i])

    # acrosacross filterType, take the difference between objEvidence and other Evidence, within only V1, include=1.
    subjects=np.unique(accuracyContainer['sub'])
    filterType=np.unique(accuracyContainer['filterType'])
    filterType=['noFilter', 'highPassRealTime', 'highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']
    a=[]
    labels=[]
    for i in range(len(filterType)):
        a.append([list(testEvidence[np.logical_and(
            np.logical_and(
                testEvidence['roi']=='V1', 
                testEvidence['filterType']==filterType[i]),
            testEvidence['include']==1.)]['obj_evidence'])])
        a.append([list(testEvidence[np.logical_and(
            np.logical_and(
                testEvidence['roi']=='V1', 
                testEvidence['filterType']==filterType[i]),
            testEvidence['include']==1.)]['otherObj_evidence'])])
        a.append([])
        labels.append(filterType[i] + ' obj_evidence')
        labels.append(filterType[i] + ' otherObj_evidence')
        labels.append('')
    bar(a,labels=labels,title='across filterType, objEvidence and other Evidence, within only V1, include=1.')




    # acrosacross filterType, take the difference between objEvidence and other Evidence, within only V1, include=1.
    subjects=np.unique(accuracyContainer['sub'])
    filterType=np.unique(accuracyContainer['filterType'])
    filterType=['noFilter', 'highPassRealTime', 'highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']
    a=[]
    labels=[]
    for i in range(len(filterType)):
        t=testEvidence[np.logical_and(
            np.logical_and(
                testEvidence['roi']=='V1', 
                testEvidence['filterType']==filterType[i]),
            testEvidence['include']==1.)]
        t=list(np.asarray(t['obj_evidence'])-np.asarray(t['otherObj_evidence']))
        a.append([t])
    bar(a,labels=filterType,title='across filterType, take the difference between objEvidence and other Evidence, within only V1, include=1.')




    # across filterType, take the difference between objEvidence and other Evidence, within only V1, include=1.
    subjects=np.unique(accuracyContainer['sub'])
    filterType=np.unique(accuracyContainer['filterType'])
    filterType=['noFilter', 'highPassRealTime', 'highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']
    a=[]
    labels=[]

    a=[]
    for i in range(len(filterType)):
        c=[]
        d=[]
        for sub in subjects:
            t=testEvidence[_and_([
                testEvidence['roi']=='V1',
                testEvidence['filterType']==filterType[i],
                testEvidence['include']==1.,
                testEvidence['sub']==sub
            ])]['obj_evidence']
            c.append(np.nanmean(t))
            d.append(np.nanmean(testEvidence[_and_([
                testEvidence['roi']=='V1',
                testEvidence['filterType']==filterType[i],
                testEvidence['include']==1.,
                testEvidence['sub']==sub
            ])]['otherObj_evidence']))
        a.append(c)
        a.append(d)
        a.append([])
        labels.append(filterType[i] + ' obj_evidence')
        labels.append(filterType[i] + ' otherObj_evidence')
        labels.append('')
    bar(a,labels=labels,title='across filterType, objEvidence and other Evidence, within only V1, include=1.')




    # accuracy: across filterType, take subject mean, within only V1, include=1.
    subjects=np.unique(accuracyContainer['sub'])
    filterType=np.unique(accuracyContainer['filterType'])
    filterType=['noFilter', 'highPassRealTime', 'highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']
    a=[]
    for sub in subjects:
        t=[list(accuracyContainer[
                _and_([
                    accuracyContainer['roi']=='V1', 
                    accuracyContainer['filterType']==filterType[i],
                    accuracyContainer['sub']==sub,
                    accuracyContainer['include']==1.
                ])]['acc']) for i in range(len(filterType))]             
        a.append(np.mean(np.asarray(t),axis=1))
    a=np.asarray(a)
    b=[a[:,i] for i in range(a.shape[1])]
    bar(b,labels=list(filterType),title='across filterType, take subject mean, within only V1, include=1.')