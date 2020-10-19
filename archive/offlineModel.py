# This model is using freesurfer ROI (V1)
# After collecting data from day1, run a freesurfer on the day1 data and define freesurfer ROIs. 
# In day two, fitted freesurfer ROI can be directly applied to the realtime dicom files.

# Alternatively, we can use searchlight to find the best predictive brain areas in day1 data, 
# use that as mask to use in day2 during realtime scan.

# I want the model to be trained in functional space of day1, not as usual register to anatomical space. This is to avoid additional transformation step.
# Designed transformational steps:
#     realtime volume to day2 template functional volume
#     then to day1 template functional volume

def offlineModel(sub='sub001',ses=1,testRun=None, FEAT=None, META=None):
	# input of this function should be the brain and behavior data 
	# output of this function should be files saved in subject data folder in current session
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
	    # Find the number of voxels that will be left given your inclusion parameter 'include'
	    return int(np.ceil(n_vox * prop))

	def get_inds(X, Y, pair, run=None):
	    '''
	    INPUT: 
	    X: brain activity
	    Y: labels
	    pair: the current working pair, e.g: bed chair
	    run: the id of the testing run, e.g. 6

	    This function extract the data for the current pair (bed chair), and train a 
	    classifier model using this. Extract the beta parameter of this model, scale 
	    it up with the mean activity of either object, as importance map. The output
	    is the importance of each voxel sorted as from least to most important for a 
	    given category.

	    '''
	    inds = {}
	    # return relative indices
	    if run: # in the case when run=6, the 6th run is left as test data, others as training data
	        trainIX = Y.index[(Y['label'].isin(pair)) & (Y['run_num'] != int(run))]
	    else: # when run is None, no testing data is used, all as training data
	        trainIX = Y.index[(Y['label'].isin(pair))]

	    # pull training and test data
	    trainX = X[trainIX]
	    trainY = Y.iloc[trainIX].label
	    
	    # train the Main classifier on 5 runs, testing on 6th
	    clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
	                             multi_class='multinomial').fit(trainX, trainY)
	    B = clf.coef_[0]  # pull betas (model paramters)

	    # retrieve only the first object, then only the second object
	    if run:  # in the case when run=6, the 6th run is left as test data, others as training data
	        obj1IX = Y.index[(Y['label'] == pair[0]) & (Y['run_num'] != int(run))] # The id of the first object
	        obj2IX = Y.index[(Y['label'] == pair[1]) & (Y['run_num'] != int(run))] # The id of the second object
	    else: # when run is None, no testing data is used, all as training data
	        obj1IX = Y.index[(Y['label'] == pair[0])]
	        obj2IX = Y.index[(Y['label'] == pair[1])]

	    # Get the average of the first object, then the second object
	    obj1X = np.mean(X[obj1IX], 0) # mean of the activity of the first object
	    obj2X = np.mean(X[obj2IX], 0) # mean of the activity of the second object

	    # Build the importance map
	    mult1X = obj1X * B # beta parameter of the trained model is the importance map scaled by mean.
	    mult2X = obj2X * B

	    # Sort these so that they are from least to most important for a given category.
	    sortmult1X = mult1X.argsort()[::-1]
	    sortmult2X = mult2X.argsort()

	    # add to a dictionary for later use
	    inds[clf.classes_[0]] = sortmult1X
	    inds[clf.classes_[1]] = sortmult2X
	             
	    return inds

	# data_dir='/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/'
	# workding_dir='/gpfs/milgram/project/turk-browne/users/kp578/realtime/Anne/'
	# files = os.listdir(data_dir+'features')
	# feats = [i for i in files if 'metadata' not in i]
	# subjects = np.unique([i.split('_')[0] for i in feats])

	# If you want to reduce the number of subjects used for testing purposes
	# subs=4
	# subjects = subjects[:subs]
	# print(subjects)

	# roi = 'V1'
	highdict = {}
	scoredict = {}

	objects = ['bed', 'bench', 'chair', 'table']
	phases = ['12', '34', '56']

	# # THIS CELL READS IN ALL OF THE PARTICIPANTS' DATA and fills into dictionary
	# FEATDICT = {}
	# METADICT = {}
	# for si, sub in enumerate(subjects[:]):
	#     print('{}/{}'.format(si+1, subs))
	#     diffs = []
	#     scores = []
	#     subcount = 0
	#     for phase in phases:
	#         _feat = np.load(data_dir+'features/{}_{}_{}_featurematrix.npy'.format(sub, roi, phase))
	#         _feat = normalize(_feat)
	#         _meta = pd.read_csv(data_dir+'features/metadata_{}_{}_{}.csv'.format(sub, roi, phase))
	#         FEAT = _feat if phase == "12" else np.vstack((FEAT, _feat))
	#         META = _meta if phase == "12" else pd.concat((META, _meta))
	#     META = META.reset_index(drop=True)

	#     assert FEAT.shape[0] == META.shape[0]
	    
	#     METADICT[sub] = META
	#     FEATDICT[sub] = FEAT
	#     clear_output(wait=True)

	# Which run to use as test data (leave as None to not have test data)
	run = testRun # this used to be 6, which means use the 6th run as the testing data and other as training data.

	# Decide on the proportion of crescent data to use for classification
	include = 1  
	allpairs = itertools.combinations(objects,2)
	
	# Iterate over all the possible target pairs of objects
	for pair in allpairs: # e.g pair is AB or AC or AD or BC or BD or CD
	    # Find the control (remaining) objects for this pair
	    altpair = other(pair) # e.g. when pair is AB, altpair is CD
	    
	    # pull sorted indices for each of the critical objects, in order of importance (low to high)
	    inds = get_inds(FEAT, META, pair, run=run)
	    
	    # Find the number of voxels that will be left given your inclusion parameter above
	    nvox = red_vox(FEAT.shape[1], include)
	    
	    for obj in pair: # e.g.obj is A or B
	        # foil = [i for i in pair if i != obj][0]
	        for altobj in altpair:
	            
	            # establish a naming convention where it is $TARGET_$CLASSIFICATION
	            # Target is the NF pair (e.g. bed/bench)
	            # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
	            naming = '{}{}_{}{}'.format(pair[0], pair[1], obj, altobj)
	            # for the pair AB, posible combinations are AB_AC AB_AD AB_BC AB_BD, the classification 
	            # is done between obj, altobj, pair is only for the purpose of organizning the loop
	            
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
	                trainX = trainX[:, obj_inds[-nvox:]] # only use the most important map
	                testX = testX[:, obj_inds[-nvox:]]
	            
	            # Train your classifier
	            clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
	                                     multi_class='multinomial').fit(trainX, trainY)
	            
	            # Save it for later use
	            savedModelFolder=f"subjects/{sub}/ses{ses}_recognition/"
	            if not os.path.isdir(savedModelFolder+'clf'):
	                os.mkdir(workding_dir+'clf')
	            joblib.dump(clf, workding_dir+'clf/{}_{}.joblib'.format(sub, naming))
	            
	            # Monitor progress by printing accuracy (only useful if you're running a test set aka when run is not None)
	            acc = clf.score(testX, testY)
	            print(naming, acc)



# ##################################################################
# ##################################################################
# ##################################################################
# ##################################################################
# ######################use the trained model#######################
# ##################################################################
# ##################################################################
# ##################################################################
# ##################################################################
# # This part should be integrated into the rtcloud framework, and finnaly send the 
# # calculated NFparam metric to the display code using 
# # WsFeedbackReceiver.startReceiverThread(rt-cloud/rtCommon/feedbackReceiver.py)


# # This function is how you would load a saved classifier for a given subject, target axis and control classifier
# clf = joblib.load(workding_dir+'clf/1206162_bedbench_bedtable.joblib') 

# # Test the classifier on a new TR (assuming X has shape [1, nvox], and Y is [label]
# # X AND Y WILL NEED REPLACED
# acc = clf.score(X, Y)

# # If running NF to try to activate table during bench for subject 0118171, you would do this prior to starting:
# # naming convention where it is $TARGET_$CLASSIFICATION
# # Target is the NF pair (e.g. bed/bench)
# # Classificationis is btw one of the targets, and a control (e.g. bed/chair, or bed/table, NOT bed/bench)
# clf1 = joblib.load(workding_dir+'clf/0118171_benchtable_tablebed.joblib') 
# clf2 = joblib.load(workding_dir+'clf/0118171_benchtable_tablechair.joblib') 

# # then do this for each TR
# s1 = clf1.score(newTR, ['table'])
# s2 = clf2.score(newTR, ['table'])
# NFparam = s1 + s2 # or an average or whatever
