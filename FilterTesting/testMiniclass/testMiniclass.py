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

def getEvidence(sub,testEvidence,METADICT=None,FEATDICT=None,filterType=None,roi="V1",include=1):
    # each testRun, each subject, each target axis, each target obj would generate one.
    META = METADICT[sub]
    FEAT = FEATDICT[sub]
    # Using the trained model, get the evidence
    testRun=6
    objects=['bed', 'bench', 'chair', 'table']
    allpairs = itertools.combinations(objects,2)
    for pair in allpairs: #pair=('bed', 'bench')
        # Find the control (remaining) objects for this pair
        altpair = other(pair) #altpair=('chair', 'table')
        for obj in pair: #obj='bed'
            # in the current target axis pair=('bed', 'bench') altpair=('chair', 'table'), display image obj='bed'
            # find the evidence for bed from the (bed chair) and (bed table) classifier

            # get the test data and seperate the test data into category obj and category other
            objIX = META.index[(META['label'].isin([obj])) & (META['run_num'] == int(testRun))]
            otherObjIX = META.index[(META['label'].isin(other(obj))) & (META['run_num'] == int(testRun))]

            obj_X=FEAT[objIX]
            obj_Y=META.iloc[objIX].label
            otherObj_X=FEAT[otherObjIX]
            otherObj_Y=META.iloc[otherObjIX].label

            model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{filterType}/'
            clf1 = joblib.load(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[0]}.joblib')
            clf2 = joblib.load(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[1]}.joblib')

            s1 = clf1.score(obj_X, obj_Y)
            s2 = clf2.score(obj_X, obj_Y)
            obj_evidence = np.mean([s1, s2])
            print('obj_evidence=',obj_evidence)

            s1 = clf1.score(otherObj_X, otherObj_Y)
            s2 = clf2.score(otherObj_X, otherObj_Y)
            otherObj_evidence = np.mean([s1, s2])
            print('otherObj_evidence=',otherObj_evidence)

            testEvidence = testEvidence.append({
                'sub':sub,
                'testRun':testRun,
                'targetAxis':pair,
                'obj':obj,
                'obj_evidence':obj_evidence,
                'otherObj_evidence':otherObj_evidence,
                'filterType':filterType,
                'include':include,
                'roi':roi
                },
                ignore_index=True)

    return testEvidence

def minimalClass(filterType = 'noFilter',testRun = 6, roi="V1",include = 1): #include is the proportion of features selected
    
    accuracyContainer = pd.DataFrame(columns=['sub','testRun','targetAxis','obj','altobj','acc','filterType','roi'])
    testEvidence = pd.DataFrame(columns=['sub','testRun','targetAxis','obj','obj_evidence','otherObj_evidence','filterType','roi'])

    working_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/FilterTesting/neurosketch_realtime_preprocess/'
    os.chdir(working_dir)

    data_dir=f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/features/{filterType}/recognition/'
    files = os.listdir(data_dir)
    feats = [i for i in files if 'metadata' not in i]
    subjects = np.unique([i.split('_')[0] for i in feats if i.split('_')[0] not in ['1121161','0112174']]) # 1121161 has a grid spacing issue and 0112174 lacks one of regressor file
    # If you want to reduce the number of subjects used for testing purposes
    subs=len(subjects)
    # subs=1
    subjects = subjects[:subs]
    print(subjects)

    highdict = {}
    scoredict = {}

    objects = ['bed', 'bench', 'chair', 'table']
    phases = ['12', '34', '56']

    # THIS CELL READS IN ALL OF THE PARTICIPANTS' DATA and fills into dictionary
    FEATDICT = {}
    METADICT = {}
    subjects_new=[]
    for si, sub in enumerate(subjects[:]):
        print('{}/{}'.format(si+1, len(subjects)))
        diffs = []
        scores = []
        subcount = 0
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

    # Which run to use as test data (leave as None to not have test data)
    subjects=subjects_new

    # Decide on the proportion of crescent data to use for classification
    
    for si,sub in enumerate(subjects):
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
                    clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                            multi_class='multinomial').fit(trainX, trainY)
                    
                    model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{include}/{roi}/{filterType}'
                    call(f"mkdir -p {model_folder}",shell=True)
                    joblib.dump(clf, model_folder +'{}_{}_{}.joblib'.format(sub, naming))
                    # Monitor progress by printing accuracy (only useful if you're running a test set)
                    acc = clf.score(testX, testY)
                    if (si+1)%10==0:
                        print(naming, acc)
                    accuracyContainer = accuracyContainer.append({
                        'sub':sub,
                        'testRun':testRun,
                        'targetAxis':pair,
                        'obj':obj,
                        'altobj':altobj,
                        'acc':acc,
                        'filterType':filterType,
                        'include':include,
                        'roi':roi
                        },
                        ignore_index=True)
    
    for sub in subjects:
        testEvidence=getEvidence(sub,testEvidence,METADICT=METADICT,FEATDICT=FEATDICT,filterType=filterType,roi=roi,include=include)
    print(accuracyContainer)
    print(testEvidence)
    accuracyContainer.to_csv(f"{model_folder}accuracy.csv")
    testEvidence.to_csv(f'{model_folder}testEvidence.csv')


include=int(sys.argv[1])
roi=sys.argv[2]
filterType=sys.argv[3]

minimalClass(filterType = include, roi=roi, include = include, testRun = 6)


# ## - to run in parallel
# # bash to submit jobs, in the folder of testMiniclass

# # testMiniclass_child.sh
# #!/bin/bash
# #SBATCH --partition=short,scavenge
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --time=3:00:00
# #SBATCH --job-name=miniClass
# #SBATCH --output=logs/miniClass-%j.out
# #SBATCH --mem-per-cpu=30G
# #SBATCH --mail-type=FAIL
# mkdir -p logs/
# module load miniconda
# source activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtAtten
# include=$1
# roi=$2
# filterType=$3
# /usr/bin/time python -u /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/FilterTesting/testMiniclass/testMiniclass.py $include $roi $filterType


# # testMiniclass_parent.py
# from glob import glob
# import os
# from subprocess import call
# for include in [0.1,0.3,0.6,0.9,1]:
#     for roi in ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']:
#         for filterType in ['noFilter','highPassRealTime','highPassBetweenRuns']:
#             minimalClass(filterType = 'noFilter',testRun = 6, roi="V1",include = 1)
#             command=f'sbatch testMiniclass_child.sh {include} {roi} {filterType}'
#             print(command)
#             # call(command, shell=True)        





# noFilter,testEvidence_noFilter=minimalClass(filterType = 'noFilter')
# noFilter.to_csv(f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/noFilter.csv')
# testEvidence_noFilter.to_csv(f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/testEvidence_noFilter.csv')

# highPassRealTime,testEvidence_highPassRealTime=minimalClass(filterType = 'highPassRealTime')
# highPassRealTime.to_csv(f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/highPassRealTime.csv')
# testEvidence_highPassRealTime.to_csv(f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/testEvidence_highPassRealTime.csv')

# highPassBetweenRuns,testEvidence_highPassBetweenRuns=minimalClass(filterType = 'highPassBetweenRuns')
# highPassBetweenRuns.to_csv(f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/highPassBetweenRuns.csv')
# testEvidence_highPassBetweenRuns.to_csv(f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/testEvidence_highPassBetweenRuns.csv')


# # load and plot data
# testEvidence_noFilter=pd.read_csv(f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/testEvidence_noFilter.csv')
# testEvidence_highPassRealTime=pd.read_csv(f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/testEvidence_highPassRealTime.csv')
# testEvidence_highPassBetweenRuns=pd.read_csv(f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/testEvidence_highPassBetweenRuns.csv')
# def bar(L,labels=None,title=None):
#     import matplotlib.pyplot as plt
#     CTEs = [np.mean(i) for i in L]
#     error = [np.std(i) for i in L]
#     x_pos = np.arange(len(labels))
#     fig, ax = plt.subplots(figsize=(10,10))
#     ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
#     ax.set_ylabel('object evidence')
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(labels)
#     ax.set_title(title)
#     ax.yaxis.grid(True)
#     plt.tight_layout()
#     plt.xticks(rotation=30)
#     plt.show()
# bar([np.asarray(testEvidence_noFilter['obj_evidence']),
#     np.asarray(testEvidence_noFilter['otherObj_evidence']),
#     [],
#     np.asarray(testEvidence_highPassRealTime['obj_evidence']),
#     np.asarray(testEvidence_highPassRealTime['otherObj_evidence']),
#     [],
#     np.asarray(testEvidence_highPassBetweenRuns['obj_evidence']),
#     np.asarray(testEvidence_highPassBetweenRuns['otherObj_evidence'])],
#     labels=[
#     'noFilter_obj',
#     'noFilter_otherObj',
#     '',
#     'highPassRealTime_obj',
#     'highPassRealTime_otherObj',
#     '',
#     'highpassOffline_obj',
#     'highpassOffline_otherObj'],title=None)

