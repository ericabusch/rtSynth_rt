# replicate neuroSketch data classification to make sure the model is trained in the 
# right way. We can transform AB AC AD BC BD CD classifiers to 4-way classifier by 
# computing the evidence (for an input X) for A B C D and output the item with the 
# largest evidence. And then calculate the testing accuracy.

# The first bullet point, you can do what we did, which is just to do the same 
# analysis that's in the paper essentially. So a 4-way classification within a 
# given phase (e.g. train on run 3 and test on run 4 and vice versa). Then we 
# can make sure that we're at least getting the same accuracy, so there isn't 
# something weird going on with the data.

# test that 4 way classifier can be trained by passing 4 classes to the classifier 
# training.

# We constructed 95% confidence intervals (CIs) for estimates of decoding accuracy 
# for each ROI by bootstrap resampling participants 10,000 times.
# How to resample across subjects?

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
from tqdm import tqdm
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def resample(L):
    L=np.asarray(L)
    sample_mean=[]
    for iter in tqdm(range(10000)):
        resampleID=np.random.choice(len(L), len(L), replace=True)
        resample_acc=L[resampleID]
        sample_mean.append(np.mean(resample_acc))
    sample_mean=np.asarray(sample_mean)
    m = np.mean(sample_mean,axis=0)
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

def strMinus(a,b):
    return [i for i in a if i!=b][0]

def listMinus(a,b):
    return [i for i in a if i!=b]

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

def minimalClass(filterType = 'noFilter',testRun = 6, roi="V1",include = 1): #include is the proportion of features selected
    
    accuracyContainer = pd.DataFrame(columns=['sub','testRun','acc','filterType','roi'])

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

    objects = ['bed', 'bench', 'chair', 'table']
    phases = ['12', '34', '56']

    # THIS CELL READS IN ALL OF THE PARTICIPANTS' DATA and fills into dictionary
    FEATDICT = {}
    METADICT = {}
    subjects_new=[]
    for si, sub in enumerate(subjects[:]):
        try:
            # print('{}/{}'.format(si+1, len(subjects)))
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

    # Decide on the proportion of crescent data to use for classification
    for si,sub in tqdm(enumerate(subjects)):
        for phase in phases:
            for curr_run in phase:
                # print('{}/{}'.format(si+1, len(subjects)))
                print(sub)
                META = METADICT[sub]
                FEAT = FEATDICT[sub]

                # This use one run as training and one run as testing
                # trainIX = META.index[(META['run_num'] == int(curr_run))]
                # testIX = META.index[(META['run_num'] == int(strMinus(phase,curr_run)))]
                # This use one run as testing and the rest as training
                trainIX = META.index[(META['run_num'] != int(curr_run))]
                testIX = META.index[(META['run_num'] == int(curr_run))]

                # pull training and test data
                trainX = FEAT[trainIX]
                testX = FEAT[testIX]
                trainY = META.iloc[trainIX].label
                testY = META.iloc[testIX].label

                # Train your classifier
                clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                        multi_class='multinomial').fit(trainX, trainY)
                
                joblib.dump(clf, model_folder + '{}_{}.joblib'.format(sub, '4WayClassifier'))

                # Monitor progress by printing accuracy (only useful if you're running a test set)
                acc = clf.score(testX, testY)
                if (si+1)%10==0:
                    print('4WayClassifier', acc)
                accuracyContainer = accuracyContainer.append({
                    'sub':sub,
                    'testRun':testRun,
                    'acc':acc,
                    'filterType':filterType,
                    'include':include,
                    'roi':roi,
                    'phase':phase,
                    'curr_run':curr_run
                    },
                    ignore_index=True)

    print(accuracyContainer)
    accuracyContainer.to_csv(f"{model_folder}4WayClassifier_accuracy.csv")

result=[]
for roi in ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']:
    # roi='V1' #sys.argv[1] ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']
    filterType='noFilter' #sys.argv[2]
    model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{roi}/{filterType}/'
    print('model_folder=',model_folder)
    call(f"mkdir -p {model_folder}",shell=True)
    minimalClass(roi=roi, filterType = filterType, testRun = 6)
    # plot
    accuracyContainer=pd.read_csv(f"{model_folder}4WayClassifier_accuracy.csv")
    m,lower,upper=bar([list(accuracyContainer['acc'])],labels=['4wayCLassifier'],title=roi)
    result.append([m,lower,upper])

# plot the summary plot
result=np.asarray(result)
result_=result.reshape(6,3)
fig, ax = plt.subplots(figsize=(10,10))
x_pos = np.arange(len(result_))
_=ax.bar(x_pos, result_[:,0], yerr=[result_[:,1],result_[:,2]], align='center', alpha=0.5, ecolor='black', capsize=10)
_=ax.set_xticks(x_pos)
labels=['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']
_=ax.set_xticklabels(labels)
_=ax.set_ylabel('accuracy')

# get 4 way classifier accuracy for wanted run
roi="V1"
filterType='noFilter'
model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{roi}/{filterType}/'
accuracyContainer=pd.read_csv(f"{model_folder}4WayClassifier_accuracy.csv")
accuracyContainer[_and_([
    accuracyContainer['sub']==int('0110171'),
    accuracyContainer['curr_run']==6
])]['acc'].iloc[0]