def minimalClass(filterType = 'noFilter'):
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

    def get_inds(X, Y, pair, run=None):
        
        inds = {}
        
        # return relative indices
        if run:
            trainIX = Y.index[(Y['label'].isin(pair)) & (Y['run_num'] != int(run))]
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
        if run:
            obj1IX = Y.index[(Y['label'] == pair[0]) & (Y['run_num'] != int(run))]
            obj2IX = Y.index[(Y['label'] == pair[1]) & (Y['run_num'] != int(run))]
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

    working_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/expScripts/recognition/neurosketch_realtime_preprocess/'
    os.chdir(working_dir)

    data_dir=f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/features/{filterType}/recognition/'
    files = os.listdir(data_dir)
    feats = [i for i in files if 'metadata' not in i]
    subjects = np.unique([i.split('_')[0] for i in feats])

    # If you want to reduce the number of subjects used for testing purposes
    subs=4
    subjects = subjects[:subs]
    print(subjects)

    roi = 'V1'
    highdict = {}
    scoredict = {}

    objects = ['bed', 'bench', 'chair', 'table']
    phases = ['12', '34', '56']

    # THIS CELL READS IN ALL OF THE PARTICIPANTS' DATA and fills into dictionary
    FEATDICT = {}
    METADICT = {}
    for si, sub in enumerate(subjects[:]):
        print('{}/{}'.format(si+1, subs))
        diffs = []
        scores = []
        subcount = 0
        for phase in phases:
            _feat = np.load(data_dir+'/{}_{}_{}_featurematrix.npy'.format(sub, roi, phase))
            _feat = normalize(_feat)
            _meta = pd.read_csv(data_dir+'/metadata_{}_{}_{}.csv'.format(sub, roi, phase))
            FEAT = _feat if phase == "12" else np.vstack((FEAT, _feat))
            META = _meta if phase == "12" else pd.concat((META, _meta))
        META = META.reset_index(drop=True)

        assert FEAT.shape[0] == META.shape[0]
        
        METADICT[sub] = META
        FEATDICT[sub] = FEAT
        clear_output(wait=True)

    # Which run to use as test data (leave as None to not have test data)
    run = 6

    # Decide on the proportion of crescent data to use for classification
    include = 1
    accuracyContainer=[]
    for sub in subjects:
        print(sub)
        META = METADICT[sub]
        FEAT = FEATDICT[sub]
        
        allpairs = itertools.combinations(objects,2)
        
        # Iterate over all the possible target pairs of objects
        for pair in allpairs:
            # Find the control (remaining) objects for this pair
            altpair = other(pair)
            
            # pull sorted indices for each of the critical objects, in order of importance (low to high)
            inds = get_inds(FEAT, META, pair, run=run)
            
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
                    if run:
                        trainIX = META.index[(META['label'].isin([obj, altobj])) & (META['run_num'] != int(run))]
                        testIX = META.index[(META['label'].isin([obj, altobj])) & (META['run_num'] == int(run))]
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
                    
                    model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/'
                    # Save it for later use
                    joblib.dump(clf, model_folder +'/{}_{}.joblib'.format(sub, naming))
                    
                    # Monitor progress by printing accuracy (only useful if you're running a test set)
                    acc = clf.score(testX, testY)
                    print(naming, acc)
    print(accuracyContainer)
    return accuracyContainer

highPassRealTime=minimalClass(filterType = 'highPassRealTime')
highPassBetweenRuns=minimalClass(filterType = 'highPassBetweenRuns')
UnscentedKalmanFilter_filter=minimalClass(filterType = 'UnscentedKalmanFilter_filter')
UnscentedKalmanFilter_smooth=minimalClass(filterType = 'UnscentedKalmanFilter_smooth')
noFilter=minimalClass(filterType = 'noFilter')

'''
highPassRealTime 
highPassBetweenRuns 
UnscentedKalmanFilter_filter # documentation: https://pykalman.github.io/
UnscentedKalmanFilter_smooth
KalmanFilter_filter
KalmanFilter_smooth
noFilter
'''
