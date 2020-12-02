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
from scipy import stats

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

def logit(p):
    return 1/(1+np.exp(-p))

def classifierEvidence(clf,X,Y): # X shape is [trials,voxelNumber], Y is ['bed', 'bed'] for example # return a 1-d array of probability
    # This function get the data X and evidence object I want to know Y, and output the trained model evidence.
    targetID=[np.where((clf.classes_==i)==True)[0][0] for i in Y]
    # Evidence1=[clf.predict_proba(X[i,:].reshape(1,-1))[0][j] for i,j in enumerate(targetID)] # logit(X*clf.coef_+clf.intercept_)
    # Evidence2=[(np.sum(X[i,:]*clf.coef_)+clf.intercept_) if j==1 else (1-(np.sum(X[i,:]*clf.coef_)+clf.intercept_)) for i,j in enumerate(targetID)] # X*clf.coef_+clf.intercept_ # np.sum((X*clf.coef_+clf.intercept_), axis=1) #logit(np.sum((X*clf.coef_[0]+clf.intercept_),axis=1)) is very close to clf.predict_proba(X), but not exactly equal
    Evidence=(np.sum(X*clf.coef_,axis=1)+clf.intercept_) if targetID[0]==1 else (1-(np.sum(X*clf.coef_,axis=1)+clf.intercept_))
    return np.asarray(Evidence)

def saveNpInDf(array):
    dataDir='./saveNpInDf/'
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)
    fileName=dataDir+str(time.time())
    np.save(fileName,array)
    return fileName

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
    # clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
    #                         multi_class='multinomial').fit(trainX, trainY)
    clf = LogisticRegression(penalty='none',solver='lbfgs', max_iter=1000, 
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

def getEvidence(sub,testEvidence,METADICT=None,FEATDICT=None,filterType=None,roi="V1",include=1,testRun=6,model_folder=None,regularization_tag=''):
    # each testRun, each subject, each target axis, each target obj would generate one.
    META = METADICT[sub]
    print('META.shape=',META.shape)
    FEAT = FEATDICT[sub]
    # Using the trained model, get the evidence
    
    objects=['bed', 'bench', 'chair', 'table']

    # allpairs = itertools.combinations(objects,2)
    allpairs = [('bed', 'chair')  , ('bench', 'table')]

    for pair in allpairs: # e.g. pair=('bed', 'bench')
        # Find the control (remaining) objects for this pair
        altpair = other(pair) # altpair=('chair', 'table')

        # obj is A
        # otherObj is B
        # altpair[0] is C
        # altpair[1] is D

        for obj in pair: # obj='bed', obj is A
            # in the current target axis pair=('bed', 'bench') altpair=('chair', 'table'), display image obj='bed'
            # find the evidence for bed from the (bed chair) and (bed table) classifier

            # get the test data and seperate the test data into category obj and category other
            otherObj=[i for i in pair if i!=obj][0] # otherObj is B
            print('otherObj=',otherObj) # This is B
            objID = META.index[(META['label'].isin([obj])) & (META['run_num'] == int(testRun))]
            otherObjID = META.index[(META['label'].isin([otherObj])) & (META['run_num'] == int(testRun))]
            
            obj_X=FEAT[objID] # A features
            otherObj_X=FEAT[otherObjID] # B features

            print(f'loading {model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[0]}_{testRun}.joblib')
            print(f'loading {model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[1]}_{testRun}.joblib')
            AC_clf = joblib.load(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[0]}_{testRun}.joblib') # AC classifier
            AD_clf = joblib.load(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[1]}_{testRun}.joblib') # AD classifier

            if include < 1:
                # This is selected features by importance
                selectedFeatures=load_obj(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[0]}_selectedFeatures_{testRun}') # AC classifier
                obj_X=obj_X[:,selectedFeatures]
                selectedFeatures=load_obj(f'{model_folder}{sub}_{pair[0]}{pair[1]}_{obj}{altpair[1]}_selectedFeatures_{testRun}') # AD classifier
                otherObj_X=otherObj_X[:,selectedFeatures]

            AC_A_evidence = classifierEvidence(AC_clf,obj_X,[obj] * obj_X.shape[0]) # AC classifier A evidence for A trials
            AD_A_evidence = classifierEvidence(AD_clf,obj_X,[obj] * obj_X.shape[0]) # AD classifier A evidence for A trials
            A_evidence_forATrials = np.mean([AC_A_evidence, AD_A_evidence],axis=0) # AC and AD classifier A evidence for A trials ; shape=(20,)

            AC_B_evidence = classifierEvidence(AC_clf,otherObj_X,[obj] * otherObj_X.shape[0]) # AC classifier A evidence for B trials
            AD_B_evidence = classifierEvidence(AD_clf,otherObj_X,[obj] * otherObj_X.shape[0]) # AD classifier A evidence for B trials
            A_evidence_forBTrials = np.mean([AC_B_evidence, AD_B_evidence],axis=0) # AC and AD classifier A evidence for B trials ; shape=(20,)
            
            # save the evidenced to testEvidence df
            testEvidence = testEvidence.append({
                'sub':sub,
                'testRun':testRun,
                'targetAxis':pair,
                'A':obj, # A
                'B':otherObj, # B
                'C':altpair[0], # C
                'D':altpair[1], # D
                
                # 'AC_A_evidence':saveNpInDf(AC_A_evidence), # AC classifier A evidence for A
                # 'AD_A_evidence':saveNpInDf(AD_A_evidence), # AD classifier A evidence for A
                # 'AC_B_evidence':saveNpInDf(AC_B_evidence), # AC classifier A evidence for B
                # 'AD_B_evidence':saveNpInDf(AD_B_evidence), # AD classifier A evidence for B

                'A_evidence_forATrials':saveNpInDf(A_evidence_forATrials),
                'A_evidence_forBTrials':saveNpInDf(A_evidence_forBTrials),
                # 'A_evidence_forBTrials_minus_A_evidence_forBTrials':np.mean(A_evidence_forATrials) - np.mean(A_evidence_forBTrials),

                'filterType':filterType,
                'include':include,
                'regularization_tag':regularization_tag,
                'roi':roi
                },
                ignore_index=True)

    return testEvidence

def minimalClass(filterType = 'noFilter',testRun = 6, roi="V1",include = 1,model_folder=None,tag='',regularization_tag=''): #include is the proportion of features selected
    
    accuracyContainer = pd.DataFrame(columns=['sub','testRun','targetAxis','acc','filterType','roi']) # 'obj','altobj',
    testEvidence = pd.DataFrame(columns=['sub','testRun','targetAxis','filterType','roi']) # 'obj','obj_evidence','otherObj_evidence',

    # working_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/FilterTesting/neurosketch_realtime_preprocess/'
    # os.chdir(working_dir)

    # data_dir=f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/features/{filterType}/recognition/'
    if filterType=='KalmanFilter_filter_analyze_voxel_by_voxel':
        data_dir=f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/features/{filterType}/recognition/{tag}/' # condition1: filter everything (including the first 56s) train and filter the Kalman at the same time.
    else:
        data_dir=f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/features/{filterType}/recognition/'

    files = os.listdir(data_dir)
    feats = [i for i in files if 'metadata' not in i]
    subjects = np.unique([i.split('_')[0] for i in feats if i.split('_')[0] not in ['1121161','0112174']]) # 1121161 has a grid spacing issue and 0112174 lacks one of regressor file
    # If you want to reduce the number of subjects used for testing purposes
    subs=len(subjects) # subs=1
    subjects = subjects[:subs]
    # subjects=['1206161', '0119173', '1201161', '1206163', '0120171', '0110171'] #['0110171', '1206161', '0120171', '1206161', '1206163'] #['1206161', '1201161', '1206163', '0110171'] #['0110171','1206161']
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
                    naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj) # can be AB_AC AB_AD AB_BC AB_BD
                    
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
                    model_path=f'{model_folder}{sub}_{naming}_{testRun}.joblib'
                





                    # if os.path.exists(model_path):
                    #     clf=joblib.load(model_path)
                    # else:
                    print('regularization_tag=',regularization_tag)
                    if regularization_tag[0:16]=='FeatureSelection':
                        print(f"penalty='none'")
                        clf = LogisticRegression(penalty='none',solver='lbfgs', max_iter=1000, 
                            multi_class='multinomial').fit(trainX, trainY)
                    else:
                        options=regularization_tag.split('_')
                        print(f"penalty={options[0]},C={np.float(options[1])},")
                        clf = LogisticRegression(penalty=options[0],C=np.float(options[1]), solver='lbfgs', max_iter=1000, 
                            multi_class='multinomial').fit(trainX, trainY)





                    joblib.dump(clf, model_path)
                    save_obj(obj_inds[-nvox:],f'{model_folder}{sub}_{naming}_selectedFeatures_{testRun}')

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
                        'roi':roi,
                        'regularization_tag':regularization_tag
                        },
                        ignore_index=True)
        # except:
        #     pass
    
    for sub in tqdm(subjects):
        # try:
        testEvidence=getEvidence(sub,testEvidence,
        METADICT=METADICT,
        FEATDICT=FEATDICT,
        filterType=filterType,
        roi=roi,
        include=include,
        regularization_tag=regularization_tag,
        testRun=testRun,
        model_folder=model_folder
        )
        # except:
        #     pass
    print('accuracyContainer=',accuracyContainer)
    print('testEvidence=',testEvidence)
    accuracyContainer.to_csv(f"{model_folder}accuracy.csv")
    testEvidence.to_csv(f'{model_folder}testEvidence.csv')
    
# condition5: Alex's new method not fitting the parameters but directly use the filter, result looks like smoothing.
regularization_tags=[
    'l2_0.001_noFeatureSelection',
    'l2_0.01_noFeatureSelection',
    'l2_1_noFeatureSelection', #inclde=1 and L2 and C=1
    'l2_10_noFeatureSelection',
    'l2_100_noFeatureSelection',

    'l1_1_noFeatureSelection',
    'l1_10_noFeatureSelection',
    'l1_100_noFeatureSelection',

    'FeatureSelection0.1', #inclue=0.1 and no L1 or L2 regularization
    'FeatureSelection0.3', #inclue=0.1 and no L1 or L2 regularization
    'FeatureSelection0.6', #inclue=0.1 and no L1 or L2 regularization
    'FeatureSelection0.9', #inclue=0.1 and no L1 or L2 regularization
    'FeatureSelection1.0', #inclue=0.1 and no L1 or L2 regularization
]

tag="condition5"

regularization_tag_id=int(sys.argv[1])    
roi=sys.argv[2]
filterType=sys.argv[3]
testRun=int(sys.argv[4])

regularization_tag=regularization_tags[regularization_tag_id]
if regularization_tag[0:16]=='FeatureSelection':
    include=np.float(regularization_tag[-3:])
else:
    include=np.float('1')

model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{include}/{roi}/{filterType}/{testRun}/{tag}/{regularization_tag}/'
print('model_folder=',model_folder)
call(f"mkdir -p {model_folder}",shell=True)
minimalClass(include = include, roi=roi, filterType = filterType, testRun = testRun,model_folder=model_folder,tag=tag,regularization_tag=regularization_tag)




## - to run in parallel
# bash to submit jobs, in the folder of testMiniclass

# # testMiniclass_child.sh
# #!/bin/bash
# #SBATCH --partition=short,scavenge,day,long,verylong
# #SBATCH --time=2:00:00
# #SBATCH --job-name=miniClass
# #SBATCH --output=logs/miniClass-%j.out
# #SBATCH --mem-per-cpu=1G
# #SBATCH --mail-type=FAIL
# mkdir -p logs/
# module load miniconda
# source activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtAtten
# regularization_tag_id=$1
# roi=$2
# filterType=$3
# testRun=$4
# /usr/bin/time python -u /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/FilterTesting/testMiniclass/testMiniclass.py $regularization_tag_id $roi $filterType $testRun


# # testMiniclass_parent.py
# from glob import glob
# import os
# from subprocess import call
# for regularization_tag_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
#     for roi in ['V1']: #, 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']:
#         for filterType in ['noFilter']: #,'highPassRealTime','highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']:
#             for testRun in [1,2,3,4,5,6]:
#                 command=f'sbatch testMiniclass_child.sh {regularization_tag_id} {roi} {filterType} {testRun}'
#                 print(command)
#             # call(command, shell=True)        


## - load and plot data
def loadPlot(tag='condition5'):

    # modules and functions
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import pdb
    import matplotlib.pyplot as plt
    import itertools
    from scipy import stats
    import warnings

    def loadNpInDf(fileName):
        main_dir='/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/FilterTesting/testMiniclass/'
        return np.load(main_dir+fileName+'.npy')

    def preloadDfnumpy(testEvidence,List=['A_evidence_forATrials','A_evidence_forBTrials']): #'AC_A_evidence','AD_A_evidence','AC_B_evidence','AD_B_evidence',
        # this function convert the dataframe cell numpy array into real numpy array, was a string pointing to a file
        warnings.filterwarnings("ignore")
        for i in range(len(testEvidence)):
            for L in List:
                testEvidence[L].iloc[i]=loadNpInDf(testEvidence[L].iloc[i])
        warnings.filterwarnings("default")
        return testEvidence

    def _and_(L):
        if len(L)==2:
            return np.logical_and(L[0],L[1])
        else:
            return np.logical_and(L[0],_and_(L[1:]))

    def resample(L):
        L=np.asarray(L).reshape(-1)
        sample_mean=[]
        for iter in range(10000):
            resampleID=np.random.choice(L.shape[0], L.shape[0], replace=True)
            resample_acc=L[resampleID]
            sample_mean.append(np.nanmean(resample_acc))
        sample_mean=np.asarray(sample_mean)
        m = np.nanmean(sample_mean,axis=0)
        upper=np.percentile(sample_mean, 97.5, axis=0)
        lower=np.percentile(sample_mean, 2.5, axis=0)
        return m,m-lower,upper-m

    def barplot_annotate_brackets(num1, num2, data, center, height,yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
        """ 
        Annotate barplot with p-values.

        :param num1: number of left bar to put bracket over
        :param num2: number of right bar to put bracket over
        :param data: string to write or number for generating asterixes
        :param center: centers of all bars (like plt.bar() input)
        :param height: heights of all bars (like plt.bar() input)
        :param yerr: yerrs of all bars (like plt.bar() input)
        :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
        :param barh: bar height in axes coordinates (0 to 1)
        :param fs: font size
        :param maxasterix: maximum number of asterixes to write (for very small p-values)
        """

        if type(data) is str:
            text = data
        else:
            # * is p < 0.05
            # ** is p < 0.005
            # *** is p < 0.0005
            text = ''
            p = .05

            while data < p:
                if len(text)>=3:
                    break
                text += '*'
                p /= 10.

                if maxasterix and len(text) == maxasterix:
                    break

            if len(text) == 0:
                text = 'n. s.'

        lx, ly = center[num1], height[num1]
        rx, ry = center[num2], height[num2]

        if yerr:
            ly += yerr[num1]
            ry += yerr[num2]

        ax_y0, ax_y1 = plt.gca().get_ylim()
        dh *= (ax_y1 - ax_y0)
        barh *= (ax_y1 - ax_y0)

        y = max(ly, ry) + dh

        barx = [lx, lx, rx, rx]
        bary = [y, y+barh, y+barh, y]
        mid = ((lx+rx)/2, y+barh)

        plt.plot(barx, bary, c='black')

        kwargs = dict(ha='center', va='bottom')
        if fs is not None:
            kwargs['fontsize'] = fs

        plt.text(*mid, text, **kwargs)
        return 
        
    def bar(LL,labels=None,title=None,pairs=None,pvalue=None):
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
        # plt.tight_layout()
        plt.xticks(rotation=30,ha='right')
        if pairs!=None:
            for pair in pairs:
                barplot_annotate_brackets(pair[0], pair[1], pvalue[pair], x_pos, m)
                m[pair[0]]+=0.1
                m[pair[1]]+=0.1
        plt.show()
        return m,lower,upper,ax

    def assertKeys(t0,t1,keys=['testRun','targetAxis','obj','otherObj']):
        # this function compare the given keys of the given two df and return true if they are exactly the same
        for key in keys:
            if not np.all(np.asarray(t1[key])==np.asarray(t0[key])):
                return False
        return True

    def concatArrayArray(c): #[array[],array[]]
        ct=[]
        List=[list(j) for j in c] # transform [array[],array[]] to [list[],list[]]
        for i in range(len(c)):
            ct=ct+List[i] # concatenate List
        return ct


    # regularization_tags=[
    #     'l2_1_noFeatureSelection', #inclde=1 and L2 and C=1
    #     'l2_10_noFeatureSelection',
    #     'l2_100_noFeatureSelection',

    #     'l1_1_noFeatureSelection',
    #     'l1_10_noFeatureSelection',
    #     'l1_100_noFeatureSelection',

    #     'FeatureSelection0.1', #inclue=0.1 and no L1 or L2 regularization
    #     'FeatureSelection0.3', #inclue=0.1 and no L1 or L2 regularization
    #     'FeatureSelection0.6', #inclue=0.1 and no L1 or L2 regularization
    #     'FeatureSelection0.9', #inclue=0.1 and no L1 or L2 regularization
    #     'FeatureSelection1.0', #inclue=0.1 and no L1 or L2 regularization
    # ]

    regularization_tags=[
        'l2_0.001_noFeatureSelection',
        'l2_0.01_noFeatureSelection',
        'l2_1_noFeatureSelection', #inclde=1 and L2 and C=1
        'l2_10_noFeatureSelection',
        'l2_100_noFeatureSelection',

        'l1_1_noFeatureSelection',
        'l1_10_noFeatureSelection',
        'l1_100_noFeatureSelection',

        'FeatureSelection0.1', #inclue=0.1 and no L1 or L2 regularization
        'FeatureSelection0.3', #inclue=0.1 and no L1 or L2 regularization
        'FeatureSelection0.6', #inclue=0.1 and no L1 or L2 regularization
        'FeatureSelection0.9', #inclue=0.1 and no L1 or L2 regularization
        'FeatureSelection1.0', #inclue=0.1 and no L1 or L2 regularization
    ]
    
    # load saved results
    accuracyContainer=[]
    testEvidence=[]
    for regularization_tag_id in range(len(regularization_tags)):
        regularization_tag=regularization_tags[regularization_tag_id]
        if regularization_tag[0:16]=='FeatureSelection':
            include=np.float(regularization_tag[-3:])
        else:
            include=np.float('1')
        for roi in ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']:
            for filterType in ['noFilter','highPassRealTime','highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']:
                for testRun in [1,2,3,4,5,6]:
                    # if filterType=='KalmanFilter_filter_analyze_voxel_by_voxel':
                    model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{include}/{roi}/{filterType}/{testRun}/{tag}/{regularization_tag}/'
                    # else:
                    #     model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{np.float(include)}/{roi}/{filterType}/{testRun}/'
                    try:
                        accuracyContainer.append(pd.read_csv(f"{model_folder}accuracy.csv"))
                        testEvidence.append(pd.read_csv(f'{model_folder}testEvidence.csv'))
                    except:
                        pass
    accuracyContainer=pd.concat(accuracyContainer, ignore_index=True)
    testEvidence=pd.concat(testEvidence, ignore_index=True)


    global filterTypes,subjects,ROIs
    filterTypes=['noFilter', 'highPassRealTime', 'highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']
    subjects=np.unique(accuracyContainer['sub'])
    ROIs=['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']


    # def evidenceAcrossFiltertypes(ROI="V1",paired_ttest=False):
    #     # construct a list where the first one is 'A_evidence_forATrials for noFilter', second is 'A_evidence_forBTrials for noFilter', third is empty, 4th is 'A_evidence_forATrials for highpass' and so on
    #     # for each element of the list, take 'A_evidence_forATrials for noFilter' for example. This is 1440*32=46080 numbers (say we have 32 subjects), each number is raw value of the 'A_evidence_forATrials for noFilter' for that subject.
    #     a=[]
    #     labels=[]
    #     for i in range(len(filterTypes)): # for each filterType, each subject has one value for A_evidence_forATrials and another value for A_evidence_forBTrials
    #         c=[]
    #         d=[]

    #         # to get one single number for A_evidence_forATrials for each subject. 
    #         # you will need to extract the corresponding conditions and conbine the data together. 
    #         for sub in subjects:
    #             t=testEvidence[_and_([ #extract
    #                 testEvidence['roi']==ROI,
    #                 testEvidence['filterType']==filterTypes[i],
    #                 testEvidence['include']==1.,
    #                 testEvidence['sub']==sub
    #             ])]
    #             t=preloadDfnumpy(t)

    #             c.append(np.asarray(list(t['A_evidence_forATrials'])).reshape(-1)) #conbine the data together
    #             d.append(np.asarray(list(t['A_evidence_forBTrials'])).reshape(-1))

    #         a.append(concatArrayArray(c))
    #         a.append(concatArrayArray(d))
    #         a.append([])
    #         labels.append(filterTypes[i] + ' A_evidence_forATrials')
    #         labels.append(filterTypes[i] + ' A_evidence_forBTrials')
    #         labels.append('')
    #     print('len of a=',[len(i) for i in a])
    #     # paired t-test
    #     objects=np.arange(4)
    #     allpairs = itertools.combinations(objects,2)
    #     pvalue={}
    #     pairs=[]
    #     for pair in allpairs:
    #         i=pair[0]
    #         j=pair[1]
    #         if paired_ttest==True:
    #             print(f"{filterTypes[i]} {filterTypes[j]} p={stats.ttest_rel(a[i*3],a[j*3])[1]}")
    #             pvalue[(i*3,j*3)]=stats.ttest_rel(a[i*3],a[j*3])[1]
    #         else:
    #             print(f"{filterTypes[i]} {filterTypes[j]} p={stats.ttest_ind(a[i*3],a[j*3])[1]}")
    #             pvalue[(i*3,j*3)]=stats.ttest_ind(a[i*3],a[j*3])[1]

    #         pairs.append((i*3,j*3))

    #     bar(a,labels=labels,title=f'raw evidence for each trial: across filterTypes, objEvidence and other Evidence, within only {ROI}, include=1. paired_ttest={paired_ttest}',pairs=pairs,pvalue=pvalue)

    #     e=[np.asarray(a[i])[~np.isnan(np.asarray(a[i]))] for i in range(len(a))]
    #     _=plt.boxplot(e)
            
    # for i in range(len(ROIs)):
    #     evidenceAcrossFiltertypes(ROI=ROIs[i])


    # def evidenceAcrossFiltertypes_meanForSub(ROI="V1",paired_ttest=True):
    #     # construct a list where the first one is 'A_evidence_forATrials for noFilter', second is 'A_evidence_forBTrials for noFilter', third is empty, 4th is 'A_evidence_forATrials for highpass' and so on
    #     # for each element of the list, take 'A_evidence_forATrials for noFilter' for example. This is 32 numbers (say we have 32 subjects), each number is mean value of the 'A_evidence_forATrials for noFilter' for that subject.

    #     # across filterType, take the difference between objEvidence and other Evidence, within only V1, include=1.
    #     filterTypes=['noFilter', 'highPassRealTime', 'highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']

    #     # I want to construct a list where the first one is 'A_evidence_forATrials for noFilter', second is 'A_evidence_forBTrials for noFilter', third is empty, 4th is 'A_evidence_forATrials for highpass' and so on
    #     # for each element of the list, take 'A_evidence_forATrials for noFilter' for example. This is 32 numbers (say we have 32 subjects), each number is the mean value of the 'A_evidence_forATrials for noFilter' for that subject.
    #     a=[]
    #     labels=[]
    #     for i in range(len(filterTypes)): # for each filterType, each subject has one value for A_evidence_forATrials and another value for A_evidence_forBTrials
    #         c=[]
    #         d=[]

    #         # to get one single number for A_evidence_forATrials for each subject. 
    #         # you will need to extract the corresponding conditions and conbine the data together. 
    #         for sub in subjects:
    #             t=testEvidence[_and_([ #extract
    #                 testEvidence['roi']==ROI,
    #                 testEvidence['filterType']==filterTypes[i],
    #                 testEvidence['include']==1.,
    #                 testEvidence['sub']==sub
    #             ])]
    #             t=preloadDfnumpy(t)

    #             c.append(np.nanmean(np.asarray(list(t['A_evidence_forATrials'])))) #conbine the data together
    #             d.append(np.nanmean(np.asarray(list(t['A_evidence_forBTrials']))))

    #         a.append(c)
    #         a.append(d)
    #         a.append([])
    #         labels.append(filterTypes[i] + ' A_evidence_forATrials')
    #         labels.append(filterTypes[i] + ' A_evidence_forBTrials')
    #         labels.append('')
    #     print('len of a = ',[len(i) for i in a])

    #     e=[np.asarray(a[i])[~np.isnan(np.asarray(a[i]))] for i in range(len(a))]
    #     _=plt.boxplot(e)

    #     # paired t-test
    #     objects=np.arange(4)
    #     allpairs = itertools.combinations(objects,2)
    #     pvalue={}
    #     pairs=[]
    #     for pair in allpairs:
    #         i=pair[0]
    #         j=pair[1]
    #         if paired_ttest==True:
    #             print(f"{filterTypes[i]} {filterTypes[j]} p={stats.ttest_rel(a[i*3],a[j*3])[1]}")
    #             pvalue[(i*3,j*3)]=stats.ttest_rel(a[i*3],a[j*3])[1]
    #         else:
    #             print(f"{filterTypes[i]} {filterTypes[j]} p={stats.ttest_ind(a[i*3],a[j*3])[1]}")
    #             pvalue[(i*3,j*3)]=stats.ttest_ind(a[i*3],a[j*3])[1]

    #         pairs.append((i*3,j*3))

    #     bar(a,labels=labels,title=f'mean evidence for each subject: across filterTypes, objEvidence and other Evidence, within only {ROI}, include=1. paired_ttest={paired_ttest}',pairs=pairs,pvalue=pvalue)

    # for i in range(len(ROIs)):
    #     evidenceAcrossFiltertypes_meanForSub(ROI=ROIs[i])
    

    # def accuracyAcrossFiltertype(ROI="V1",paired_ttest=False):
    #     # accuracy: across filterType, take subject mean, within only V1, include=1.
        
    #     # I want to construction a list whose 1st element is the accuracy for noFilter, 2nd for highpass and so on.
    #     # each element is 32 numbers for 32 subjects. each number is the mean accuracy for that subject.
    #     a=[]
    #     for i in range(len(filterTypes)):
    #         b=[]
    #         for sub in tqdm(subjects):
    #             try:
    #                 b.append(np.mean(accuracyContainer[
    #                         _and_([
    #                             accuracyContainer['roi']==ROI, 
    #                             accuracyContainer['filterType']==filterTypes[i],
    #                             accuracyContainer['sub']==int(sub),
    #                             accuracyContainer['include']==1.
    #                         ])]['acc']))
    #             except:
    #                 pass
    #         a.append(np.asarray(b))
    #     # bar(a,labels=list(filterTypes),title=f'accuracy: across filterTypes, within only {ROI}, include=1.')
    #     e=[np.asarray(a[i])[~np.isnan(np.asarray(a[i]))] for i in range(len(a))]
    #     _=plt.boxplot(e)

    #     # paired t-test
    #     objects=np.arange(4)
    #     allpairs = itertools.combinations(objects,2)
    #     pvalue={}
    #     pairs=[]
    #     for pair in allpairs:
    #         i=pair[0]
    #         j=pair[1]
    #         if paired_ttest==True:
    #             print(f"{filterTypes[i]} {filterTypes[j]} p={stats.ttest_rel(a[i],a[j])[1]}")
    #             pvalue[(i,j)]=stats.ttest_rel(a[i],a[j])[1]
    #         else:
    #             print(f"{filterTypes[i]} {filterTypes[j]} p={stats.ttest_ind(a[i],a[j])[1]}")
    #             pvalue[(i,j)]=stats.ttest_ind(a[i],a[j])[1]

    #         pairs.append((i,j))
    #     bar(a,labels=list(filterTypes),title=f'accuracy: across filterTypes, within only {ROI}, include=1.  paired_ttest={paired_ttest}',pairs=pairs,pvalue=pvalue)

    # for i in range(len(ROIs)):
    #     accuracyAcrossFiltertype(ROI=ROIs[i])


    def accuracyIncludes(ROI="V1"):
        # compare between includes using accuracy
        # I want to construct a comparison between different includes by having includes
        filterType='noFilter'
        a=[]
        for regularization_tag_id in range(len(regularization_tags)):
            regularization_tag=regularization_tags[regularization_tag_id]
            b=[]
            for sub in tqdm(subjects):
                try:
                    b.append(np.mean(accuracyContainer[
                            _and_([
                                accuracyContainer['roi']==ROI, 
                                accuracyContainer['filterType']==filterType,
                                accuracyContainer['sub']==int(sub),
                                accuracyContainer['regularization_tag']==regularization_tag
                            ])]['acc']))
                except:
                    pass
            a.append(np.asarray(b))
        bar(a,labels=list(regularization_tags),title=f'accuracy: across include, filterType = {filterType}, within only {ROI}.')
        _=plt.figure()
        e=[np.asarray(a[i])[~np.isnan(np.asarray(a[i]))] for i in range(len(a))]
        _=plt.boxplot(e)

    for i in range(len(ROIs)):
        accuracyIncludes(ROI=ROIs[i])


    def evidenceIncludes(ROI="V1"): # filtering the features would often increase the performance.
        # compare between includes using evidence
        # I want to construct a comparison between different includes by having includes
        filterType='noFilter'
        a=[]
        for regularization_tag_id in range(len(regularization_tags)):
            regularization_tag=regularization_tags[regularization_tag_id]
            b=[]
            for sub in tqdm(subjects):
                t=testEvidence[_and_([ #extract
                    testEvidence['roi']==ROI,
                    testEvidence['filterType']==filterType,
                    testEvidence['regularization_tag']==regularization_tag,
                    testEvidence['sub']==int(sub)
                ])]
                t=preloadDfnumpy(t)
                b.append(np.nanmean(np.asarray(list(t['A_evidence_forATrials']))))
            a.append(np.asarray(b))
        bar(a,labels=list(regularization_tags),title=f'evidence across include, filterType = {filterType}, within only {ROI}.')
        _=plt.figure()
        e=[np.asarray(a[i])[~np.isnan(np.asarray(a[i]))] for i in range(len(a))]
        _=plt.boxplot(e)

    for i in range(len(ROIs)):
        evidenceIncludes(ROI=ROIs[i])
