# This script trains a model using the data from recognition session
# This code is done writing but not tested, I should test it and save the models
def minimalClass(sub='pilot_sub001',ses=1):
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

    from IPython.display import clear_output

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
        main_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/'
    else:
        main_dir='/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtcloud_kp/'

    working_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/expScripts/recognition/recognitionDataAnalysis/'
    os.chdir(working_dir)

    # data_dir=f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/features/{filterType}/recognition/'

    # files = os.listdir(data_dir)
    # feats = [i for i in files if 'metadata' not in i]
    # subjects = np.unique([i.split('_')[0] for i in feats])

    # # If you want to reduce the number of subjects used for testing purposes
    # subs=4
    # subjects = subjects[:subs]
    # print(subjects)

    roi = 'V1'
    # highdict = {}
    # scoredict = {}

    objects = ['bed', 'bench', 'chair', 'table']
    # phases = ['12', '34', '56'] #number of runs in day1recognition run
    runs=['1'] #normally this would be np.arange(1,9)

    for run in runs:
        # read in brain and behavior data
        brain_data_path=main_dir+f'subjects/{sub}/ses{ses}_recognition/run{run}/{sub}_{run}_preprocessed_brainData.npy'
        t=np.load(brain_data_path)
        brain_data=t if run=='1' else np.concatenate((brain_data,t), axis=0)

        behav_data_path=main_dir+f'subjects/{sub}/ses{ses}_recognition/run{run}/{sub}_{run}_preprocessed_behavData.csv'
        t=pd.read_csv(behav_data_path)
        behav_data=t if run=='1' else pd.concat([behav_data,t])

    FEAT=brain_data
    META=behav_data

    # convert iterm colume to label colume
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
    # testRun = None
    testRun = 2
    META['run_num'].iloc[:5]=2

    # Decide on the proportion of crescent data to use for classification
    include = 1
    accuracyContainer=[]


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
                
                model_folder = f'/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/clf/'
                # Save it for later use
                joblib.dump(clf, model_folder +'/{}_{}.joblib'.format(sub, naming))
                
                # Monitor progress by printing accuracy (only useful if you're running a test set)
                acc = clf.score(testX, testY)
                print(naming, acc)

minimalClass(sub='pilot_sub001',ses=1)
