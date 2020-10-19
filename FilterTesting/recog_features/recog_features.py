# recog_features.py This is copied from https://github.com/cogtoolslab/neurosketch/blob/master/analysis/preprocessing/prototype/link/scripts/recog_features.py
# This code is shared by Jeff as the preprocessing code for the miniclass model training

# The purpose of this code is to generate the input of the offline model training aka 
# the brain data maxtrix M x N (M time points, N voxels)and the metadata which is the 
# labels with the length of M time points.

'''
The "input" of this code is 
    regressor file: recog_reg 
        which looks like three colum regressors that are used in FEAT, each object (ABCD) has a seperate regressor file?
    raw brain data: 

    where is the behavior data? at what time is each image presented? That would be the regressor file
'''

def filtering(timeseries=None,filterType='highPassRealTime'): 
    '''
    filterType can be 
        highPassRealTime 
        highPassBetweenRuns 
        UnscentedKalmanFilter_filter # documentation: https://pykalman.github.io/
        UnscentedKalmanFilter_smooth
        KalmanFilter_filter
        KalmanFilter_smooth
        noFilter
    '''
    import numpy as np    
    import sys
    sys.path.append('/gpfs/milgram/scratch60/turk-browne/kp578/rtAttenPenn_cloud/rtAtten')
    timeseries=timeseries.astype(np.float)
    oldShape=timeseries.shape
    timeseries=timeseries.reshape(timeseries.shape[0],-1)
    if filterType == 'highPassRealTime':
        # from highpassFunc import highPassRealTime, highPassBetweenRuns
        from highpass import highpass
        def highPassRealTime(A_matrix, TR, cutoff):
            full_matrix = np.transpose(highpass(np.transpose(A_matrix), cutoff/(2*TR), True))
            return full_matrix[-1, :]

        filtered_timeseries=[]
        for currTR in range(timeseries.shape[0]):
            filtered_timeseries.append(highPassRealTime(timeseries[:(currTR+1)],1.5,56))
        filtered_timeseries = np.asarray(filtered_timeseries)
    elif filterType == 'highPassBetweenRuns':
        # from highpassFunc import highPassRealTime, highPassBetweenRuns
        from highpass import highpass
        def highPassBetweenRuns(A_matrix, TR, cutoff):
            return np.transpose(highpass(np.transpose(A_matrix), cutoff/(2*TR), False))

        filtered_timeseries = highPassBetweenRuns(timeseries,1.5,56)
    elif filterType == 'UnscentedKalmanFilter_filter' :
        from pykalman import UnscentedKalmanFilter
        ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, observation_covariance=0.1)
        filtered_timeseries=np.zeros(timeseries.shape)
        for curr_voxel in range(timeseries.shape[1]):
            (filtered_timeseries_state_means, filtered_timeseries_state_covariances) = ukf.filter(timeseries[:,curr_voxel])
            filtered_timeseries[:,curr_voxel] = filtered_timeseries_state_means.reshape(-1)
    elif filterType == 'UnscentedKalmanFilter_smooth':
        from pykalman import UnscentedKalmanFilter
        ukf = UnscentedKalmanFilter(lambda x, w: x + np.sin(w), lambda x, v: x + v, observation_covariance=0.1)
        filtered_timeseries=np.zeros(timeseries.shape)
        for curr_voxel in range(timeseries.shape[1]):
            (smoothed_state_means, smoothed_state_covariances) = ukf.smooth(data)
            filtered_timeseries[:,curr_voxel] = smoothed_state_means.reshape(-1)
    # elif filterType == 'KalmanFilter_filter':
    #     from pykalman import KalmanFilter
    #     kf = KalmanFilter(transition_matrices = None, observation_matrices = None)
    #     filtered_timeseries=np.zeros(timeseries.shape)
    #     for curr_voxel in range(timeseries.shape[1]):
    #         measurements = np.asarray(timeseries[:,curr_voxel]) 
    #         kf = kf.em(measurements, n_iter=5)
    #         (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    #         filtered_timeseries[:,curr_voxel] = filtered_state_means.reshape(-1)
    elif filterType == 'KalmanFilter_filter':
        from pykalman import KalmanFilter
        kf = KalmanFilter(n_dim_obs = timeseries.shape[1], n_dim_state = timeseries.shape[1], 
                              observation_matrices = sparse.eye(timeseries.shape[1]), 
                              observation_offsets = np.zeros(timeseries.shape[1]),
                              observation_covariance = sparse.eye(timeseries.shape[1]))
        kf = kf.em(timeseries, run, n_iter=10)
        (filtered_state_means, filtered_state_covariances) = kf.filter(timeseries)
        filtered_timeseries = filtered_state_means.reshape(-1)


    elif filterType == 'KalmanFilter_smooth':
        from pykalman import KalmanFilter
        kf = KalmanFilter(transition_matrices = None, observation_matrices = None)
        filtered_timeseries=np.zeros(timeseries.shape)
        for curr_voxel in range(timeseries.shape[1]):
            measurements = np.asarray(timeseries[:,curr_voxel]) 
            kf = kf.em(measurements, n_iter=5)
            (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
            filtered_timeseries[:,curr_voxel] = smoothed_state_means.reshape(-1)
    elif filterType == 'noFilter':
        filtered_timeseries = timeseries
    else:
        raise Exception('filterType wrong')

    filtered_timeseries=filtered_timeseries.reshape(oldShape)
    return filtered_timeseries


def recog_features(subject='0110171',filterType = 'highPassBetweenRuns'):
    import os
    import sys
    import numpy as np
    import pandas as pd
    import nibabel as nib

    os.chdir('/gpfs/milgram/scratch60/turk-browne/kp578/rtAttenPenn_cloud/rtAtten/')
    subject = subject.split('_')[0]
    proj_dir = '/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/'
    dat_type = 'recog'
    data_dir = os.path.abspath(os.path.join(proj_dir,'features')) # this is the output folder where output features are saved

    # INPUT: brain data
    feature_dir = os.path.abspath(os.path.join(data_dir, filterType)) #This is the output folder, where the output features are saved
    if not os.path.isdir(feature_dir):
        os.mkdir(feature_dir)
    filt_func = os.path.abspath(os.path.join(proj_dir, \
        'subjects/{}_neurosketch/data/nifti/realtime_preprocessed',
        '{}_neurosketch_recognition_run_{}.nii.gz')) # input, the continuous time series, brain data.
    recog_reg = os.path.abspath(os.path.join(proj_dir, \
        'subjects/{}_neurosketch/regressor/run_{}/{}.txt')) # this is regressor file

    roi_dir = os.path.abspath(os.path.join(proj_dir,'subjects/{}_neurosketch/analysis/firstlevel/rois'))

    # /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/surfroi/0110171_neurosketch_V1.nii.gz    88*128*128
    # /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/data/nifti/0110171_neurosketch_anat_mprage_brain.nii.gz.   256*256*176
    # /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/analysis/firstlevel/rois/V1_func_run_1.nii.gz 94*94*72 #note V1_func_run_{1~6}.nii.gz are the same
    # /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/data/nifti/0110171_neurosketch_recognition_run_1.nii.gz 94*94*72

    out_dir = os.path.abspath(os.path.join(feature_dir, 'recognition')) 
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    objects = ['bed', 'bench', 'chair', 'table']
    roi_list_masks = ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo'] #['V1', 'V2','LOC_FS','IT_FS','fusiform_FS','parahippo_FS','PRC_FS','ento_FS','hipp_FS','V1Draw', 'V2Draw', 'LOCDraw', 'ParietalDraw']
    roi_list_names = ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo'] #['V1','V2','LOC','IT','fusiform','parahippo','PRC','ento','hipp', 'V1Draw', 'V2Draw', 'LOCDraw', 'ParietalDraw']

    for curr_phase,phase in enumerate(['12', '34', '56']):
        # initialize data columns
        subj = [subject] * 160
        label = []
        run_num = [phase[0]]*80 + [phase[1]]*80
        TR_num = []

        for rn, run in enumerate(phase): # phase is '12' or '34' or '56'
            print('creating features for run {}'.format(run))
            # load subject's time series for this run
            timeseries = nib.load(filt_func.format(subject, subject, run))
            timeseries = timeseries.get_data().transpose((3, 0, 1, 2))
            
            timeseries = filtering(timeseries=timeseries,filterType=filterType)
            # use information in regressor/run_x folder to make hasImage vector
            # associated TR is just the hasImage index, converted to a float
            Onsets = [0]*240
            for obj in objects:
                with open(recog_reg.format(subject, run, obj)) as f:
                    times = [line.split(' ')[0] for line in f.read().split('\n')[:-1]]
                    for t in times:
                        TR = int(float(t)/1.5)
                        Onsets[TR] = obj

            # wherever hasImage, we want the features, aka only keep the features when there 
            # is an image shown
            features = [timeseries[n+3] for n, onset in enumerate(Onsets) if onset != 0] # move the timeseries by 3TRs, only keep the features when an image is shown
            labels = [label for label in Onsets if label != 0]
            FEATURES = np.array(features) if rn == 0 else np.vstack((FEATURES, np.array(features)))
            LABELS = labels if rn == 0 else LABELS + labels

        if curr_phase==0:
            FEATURES_oldShape=FEATURES.shape
        else:
            assert np.allclose(FEATURES.shape,FEATURES_oldShape),"FEATURES shape not match"

        np.save('{}/{}_{}_featurematrix.npy'.format(out_dir, subject, phase), FEATURES)
        
        for roi, roiname in zip(roi_list_masks, roi_list_names):
            # mask = nib.load('{}/{}.nii.gz'.format(roi_dir.format(subject), roi))
            mask = nib.load('{}/{}_func_combined_{}_binarized.nii.gz'.format(roi_dir.format(subject), roi,'12')) #here the phase is fixed to '12' because I found that different phases mask are different, e.g. fslview_deprecated /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/analysis/firstlevel/rois/V1_func_combined_12_binarized.nii.gz /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/analysis/firstlevel/rois/V1_func_combined_34_binarized.nii.gz are not the same!
            maskDat = mask.get_data()
            masked = FEATURES[:, maskDat == 1]
            np.save('{}/{}_{}_{}_featurematrix.npy'.format(out_dir, subject, roiname, phase), masked)
            
            ## metadata
            x = pd.DataFrame([subj, LABELS, run_num, TR_num]) # lists of the same length
            x = x.transpose()
            x.columns = ['subj','label','run_num', 'TR_num']
            x.to_csv('{}/metadata_{}_{}_{}.csv'.format(out_dir, subject, roiname, phase))

from glob import glob
import os
from subprocess import call
import sys

#installing rtAtten is very simple, just `conda env create -f environment.yml ; source activate rtAtten ; python setup.py install`
working_dir='/gpfs/milgram/scratch60/turk-browne/kp578/rtAttenPenn_cloud/rtAtten/' 
os.chdir(working_dir)
print('pwd=',os.getcwd())
print('CONDA_DEFAULT_ENV=',os.environ['CONDA_DEFAULT_ENV'])

sub=sys.argv[1]
filterType=sys.argv[2]
print('sub=',sub)
print('filterType=',filterType)

complete=f"{working_dir}complete/{sub}{filterType}.complete"
if not os.path.isdir(f"{working_dir}complete/"):
    os.mkdir(f"{working_dir}complete/")
if not os.path.exists(complete):
    recog_features(subject=sub, filterType = filterType)
    call(f"touch {complete}",shell=True)


# ## - to run all the subjects
# # bash to submit jobs, in the folder of recognition code

# # recog_features_child.sh
# #!/bin/bash
# #SBATCH --partition=short,scavenge
# #SBATCH --job-name rt_sketch
# #SBATCH --time=3:00:00
# #SBATCH --output=logs/rt_sketch-%j.out
# #SBATCH --mem=50g
# #SBATCH --mail-type=FAIL
# module load miniconda
# source activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtAtten
# sub=$1
# filters=$2
# /usr/bin/time python -u /gpfs/milgram/project/turk-browne/projects/rtcloud_kp/FilterTesting/recog_features/recog_features.py $sub $filters


# # recog_features_parent.py
# from glob import glob
# import os
# from subprocess import call
# subject_dir='/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/'
# subjects=glob(subject_dir+'*_neurosketch')
# subjects=[sub.split('/')[-1] for sub in subjects if sub.split('/')[-1][0]!='_']
# filters=[
# #'UnscentedKalmanFilter_filter', #takes too long
# #'UnscentedKalmanFilter_smooth', #takes too long
# # 'KalmanFilter_filter', #takes too long if done seperately for each voxel, takes too much memory if processing everything at the same time.
# # 'KalmanFilter_smooth',
# 'noFilter',
# 'highPassRealTime',
# 'highPassBetweenRuns' 
# ]
# for sub in subjects:
#     for curr_filter in filters:
#         command=f'sbatch recog_features_child.sh {sub} {curr_filter}'
#         print(command)
#         # call(command, shell=True)
