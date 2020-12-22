from glob import glob
import os
from subprocess import call
import subprocess
import sys
import numpy as np
import nibabel as nib
import ray
from pykalman import KalmanFilter
from time import time
from pathlib import Path
import psutil
import gc
import pandas as pd

ray.init()
print('CONDA_DEFAULT_ENV=',os.environ['CONDA_DEFAULT_ENV'])

process = subprocess.Popen(['hostname'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
cluster, err = process.communicate()

assert "filter" in sys.argv[1], "first sys.arg must name the type of filtering being done"
# assert int(sys.argv[2]) > 10000, "second sys.arg must be the subject number"
# assert sys.argv[3] in ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo'], "third sys.arg must be ROI"
if b'milgram' in cluster: # yale cluster
    subject_dir='/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/'
    working_dir="/gpfs/milgram/project/turk-browne/projects/rtSynth/alex_scratch/realtime"
    proj_dir = '/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/'
    roi_dir = os.path.abspath(os.path.join(proj_dir,'subjects/{}_neurosketch/analysis/firstlevel/rois'))
    output_dir = f"/gpfs/milgram/scratch60/turk-browne/an633/{sys.argv[1]}"
elif b'spock' in cluster: # princeton cluster
    subject_dir='/jukebox/norman/datasets/sketchloop02/subjects/'
    working_dir="/usr/people/qanguyen/rtSynth/alex_scratch/realtime"
    proj_dir = '/jukebox/norman/datasets/sketchloop02/'
    roi_dir = os.path.abspath(os.path.join(proj_dir,'subjects/{}_neurosketch/analysis/firstlevel/rois'))
    output_dir = ""

subjects=glob(subject_dir+'*_neurosketch')
subjects=[sub.split('/')[-1] for sub in subjects]
print("subjects", subjects)

subjects_batch_1 = ['0110171', '0110172', '0111171', '0112171', '0112172', '0112173'] #, '0112174'] broken
subjects_batch_2 = ['0113171', '0115172', '0115174', '0117171', '0118171', '0118172', '0119171', '0119172', '0119173', '0119174', '0120171']
subjects_batch_3 = ['0120172', '0120173', '0123171', '0123173', '0124171', '0125171']
# subjects_batch_4 = ['0125172', '1121161', '1130161', '1201161', '1202161', '1203161']
subjects_batch_4 = ['0125172', '1130161', '1201161', '1202161', '1203161'] # no subject  1121161
subjects_batch_5 = ['1206162', '1206163', '1207162']
subjects = subjects_batch_1 + subjects_batch_2 + subjects_batch_3 + subjects_batch_4 + subjects_batch_5


filterType= 'KalmanFilter_filter_analyze_voxel_by_voxel'
# either 'highPassRealTime' or 'highPassBetweenRuns'
# or 'UnscentedKalmanFilter_filter' or 'UnscentedKalmanFilter_smooth'
# or 'KalmanFilter_filter' or 'KalmanFilter_smooth' or 'noFilter'
# print('sub=',sub)
print('filterType=',filterType , flush=True)


f = open(f"{working_dir}/log2.txt", "w+")
def printwrite(*strings, writefile = f):
    print(*strings)
    strings = [str(i) for i in list(strings)]
    strings = " ".join(list(strings))
    writefile.write(strings + "\n")
    writefile.flush()



@ray.remote(num_cpus = 0.1)
def kalman_filter_voxel_filter_no_EM(measurement, curr_voxel, transition_covariance_param):
    print("transition_covariance_param ", transition_covariance_param)
#     print("float(transition_covariance_param) * np.eye(measurement[1]", (float(transition_covariance_param) * np.eye(measurement[1])).shape)
#     print("(transition_covariance_param) * np.diag(measurement[0])", ((transition_covariance_param) * np.diag(measurement[0])).shape)
#     start_time = time()
    kf = KalmanFilter(n_dim_state = measurement.shape[1], n_dim_obs = measurement.shape[1],
                          observation_matrices = np.eye(measurement.shape[1]),
                          observation_offsets = np.zeros(measurement.shape[1]),
                          initial_state_mean = measurement[0, :],
                      initial_state_covariance = np.eye(measurement.shape[1]),
                      transition_matrices = np.eye(measurement.shape[1]),
                      transition_offsets =   np.zeros(measurement.shape[1]),
                       # transition_covariance = (transition_covariance_param) * np.diag(measurement[0]),
                      transition_covariance = float(transition_covariance_param) * np.eye(measurement.shape[1]),
                       observation_covariance = 600 * np.eye(measurement.shape[1])
                          )


    filtered_timeseries = [measurement[:2, :].copy()] # do not filter in the first 37 TRs (first 56 seconds)
    filtered_state_means, filtered_state_covariances = kf.filter(measurement[:2])
    filtered_state_mean, filtered_state_covariance = [filtered_state_means[-1,:], filtered_state_covariances[-1,:]]
    for t in range(3, measurement.shape[0] + 1):
        filtered_state_mean, filtered_state_covariance = kf.filter_update(filtered_state_mean, filtered_state_covariance, observation=measurement[t-1])
        filtered_timeseries.append(filtered_state_mean)

    filtered_timeseries = np.vstack(filtered_timeseries)

    return filtered_timeseries

kalman_filtering_function = kalman_filter_voxel_filter_no_EM



def filtering(run, timeseries=None,filterType='highPassRealTime', kalman_filtering_function = kalman_filtering_function, transition_cov = None):
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

    timeseries=timeseries.astype(np.float)
    oldShape=timeseries.shape
    timeseries=timeseries.reshape(timeseries.shape[0],-1)
    if filterType == 'KalmanFilter_filter_analyze_voxel_by_voxel':

        filtered_timeseries=np.zeros(timeseries.shape)
        print("time series shape", timeseries.shape, flush=True)
        futures = []
        state_dim = 50
#         state_dim = timeseries.shape[1]
        for curr_voxel in range(0, timeseries.shape[1], state_dim):
        # for curr_voxel in range(2):
            measurements = np.asarray(timeseries[:,curr_voxel:(curr_voxel + state_dim)])
#             print("measurements", measurements.reshape(-1, state_dim).shape)
#             raise ValueError
#
            futures.append(kalman_filtering_function.remote(measurements, curr_voxel, transition_cov))
            # break
            # kalman_filter_voxel(measurements)

        results = ray.get(futures)
        filtered_timeseries = np.ma.filled(np.hstack(results)) # transpose because results are organized by [voxel, time]
        print("filtered_timeseries shape", filtered_timeseries.shape)
        print("filtered_timeseries.T", filtered_timeseries.T[0])
        print("non filtered_timeseries.T", timeseries.T[0])

        # for _ in range(20):
        #     plt.figure(figsize=(10,10))
        #     i,j=[4+2*_,5+2*_]
        #     plt.plot(timeseries.T[i])
        #     plt.plot(filtered_timeseries.T[i])
        #     plt.plot(timeseries.T[j])
        #     plt.plot(filtered_timeseries.T[j])
        #     plt.title(f"{i},{j}")
        #     plt.show()

    elif filterType == 'noFilter':
        filtered_timeseries = timeseries
    else:
        raise Exception('filterType wrong')

    filtered_timeseries=filtered_timeseries.reshape(oldShape)
    return filtered_timeseries

from memory_profiler import profile

@profile(precision=4, stream = open("/gpfs/milgram/project/turk-browne/projects/rtSynth/alex_scratch/realtime/memory.txt", "w+"))
def filter_memory_wrapper(run, roi_timeseries, filterType, transition_cov):
    return filtering(run, timeseries=roi_timeseries,filterType=filterType, transition_cov=transition_cov)

def recog_features(subject='0110171',filterType = 'highPassBetweenRuns', roi_list_masks = None, roi_list_names = None, transition_cov = None,
                    start_voxel= None, end_voxel= None,
                    proj_dir = proj_dir, roi_dir = roi_dir, output_dir = output_dir):





    # os.chdir('/gpfs/milgram/scratch60/turk-browne/kp578/rtAttenPenn_cloud/rtAtten/')

    # subject = '0110171' #sys.argv[1] # This is the subject name

    subject = subject.split('_')[0]
    print("subject=", subject)

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

    #roi_list_masks = ['fusiform', 'IT', 'LOC', 'occitemp', 'parahippo'] #['V1', 'V2','LOC_FS','IT_FS','fusiform_FS','parahippo_FS','PRC_FS','ento_FS','hipp_FS','V1Draw', 'V2Draw', 'LOCDraw', 'ParietalDraw']
    #roi_list_names = ['fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']
    # roi_list_masks = ['V1']
    # roi_list_names = ['V1']
    for curr_phase,phase in enumerate(['12', '34', '56']):
        # initialize data columns
        subj = [subject] * 160
        label = []
        run_num = [phase[0]]*80 + [phase[1]]*80
        TR_num = []

        for roi, roiname in zip(roi_list_masks[:9], roi_list_names[:9]):
            for rn, run in enumerate(phase): # phase is '12' or '34' or '56'
                print('creating features for run {}'.format(run),   flush=True)
                # load subject's time series for this run
                print("Load time series for this run from", filt_func.format(subject, subject, run),  flush=True)
                timeseries = nib.load(filt_func.format(subject, subject, run))
                timeseries = timeseries.get_fdata().transpose((3, 0, 1, 2))

                if not os.path.isdir(f"{output_dir}/{roi}"):
                    Path(f"{output_dir}/{roi}").mkdir(parents=True, exist_ok=True)
                # only filter voxels in ROI of interest
                mask = nib.load('{}/{}_func_combined_{}_binarized.nii.gz'.format(roi_dir.format(subject), roi, '12')) #here the phase is fixed to '12' because I found that different phases mask are different, e.g. fslview_deprecated /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/analysis/firstlevel/rois/V1_func_combined_12_binarized.nii.gz /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/analysis/firstlevel/rois/V1_func_combined_34_binarized.nii.gz are not the same!
                maskDat = mask.get_fdata()
                roi_timeseries = timeseries[:, maskDat == 1]#[:, (start_voxel) : (end_voxel)]

                # Skip processing this run if output file already exists (and has the correct dimensionality)
                # output_file_name = '{}/{}/{}/{}_Kalman_filter_{}_{}_featurematrix_voxel_{}_to_{}.npy'.format(output_dir, roi, subject, roi, subject, run, start_voxel, end_voxel)
                output_file_name = '{}/{}/{}_Kalman_filter_{}_{}_featurematrix.npy'.format(output_dir, roi, roi, subject, run)
                if os.path.exists(output_file_name):
                    already_existing_filtered_series = np.load(output_file_name)
                    if already_existing_filtered_series.shape == roi_timeseries.shape:
                        print("Already have", output_file_name)
                        continue

                # start_time = time()
                roi_timeseries = filter_memory_wrapper(run, roi_timeseries, filterType, transition_cov)
                # end_time = time()
                # printwrite(f"Subject {subject} {roi} Processing {roi_timeseries.shape[-1]} voxels for Run {run} in {end_time - start_time} seconds")
                # np.save(output_file_name, roi_timeseries)

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
                features = [roi_timeseries[n+3] for n, onset in enumerate(Onsets) if onset != 0] # move the roi_timeseries by 3TRs, only keep the features when an image is shown
                labels = [label for label in Onsets if label != 0]
                FEATURES = np.array(features) if rn == 0 else np.vstack((FEATURES, np.array(features)))
                LABELS = labels if rn == 0 else LABELS + labels

            # if curr_phase==0:
            #     FEATURES_oldShape=FEATURES.shape
            # else:
            #     print("FEATURES_oldShape", FEATURES_oldShape,"FEATURES.shape", FEATURES.shape, "run", run, "roi", roi)
            #     assert np.allclose(FEATURES.shape,FEATURES_oldShape),"FEATURES shape not match"
            print("FEATURES.shape", FEATURES.shape, "run", run, "roi", roi)
            np.save('{}/{}/{}_{}_{}_featurematrix.npy'.format(output_dir, roi, subject, roi, phase), FEATURES)

            # for roi, roiname in zip(roi_list_masks[:9], roi_list_names[:9]):
            # mask = nib.load('{}/{}.nii.gz'.format(roi_dir.format(subject), roi))
            # mask = nib.load('{}/{}_func_combined_{}_binarized.nii.gz'.format(roi_dir.format(subject), roi,'12')) #here the phase is fixed to '12' because I found that different phases mask are different, e.g. fslview_deprecated /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/analysis/firstlevel/rois/V1_func_combined_12_binarized.nii.gz /gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/analysis/firstlevel/rois/V1_func_combined_34_binarized.nii.gz are not the same!
            # maskDat = mask.get_data()
            # print("FEATURES", FEATURES.shape)
            # print("maskDat", maskDat )
            # masked = FEATURES[:, maskDat == 1]
            # np.save('{}/{}/{}_{}_{}_featurematrix.npy'.format(output_dir, roi,  subject, roiname, phase), masked)

            ## metadata
            x = pd.DataFrame([subj, LABELS, run_num, TR_num]) # lists of the same length
            x = x.transpose()
            x.columns = ['subj','label','run_num', 'TR_num']
            x.to_csv('{}/{}/metadata_{}_{}_{}.csv'.format(output_dir, roi, subject, roiname, phase))



if __name__ == '__main__':
    transition_cov = float(sys.argv[2])
    print("transition covariance" , (transition_cov))
    try:
        # Your normal block of code
        # for sub in subjectssqueue [::-1]:
        for sub in subjects:
            recog_features(
                # subject=sys.argv[2],
                subject= sub,
#                 roi_list_masks=[roi],
#                 roi_list_names=[roi],
                # start_voxel=int(sys.argv[4]),
                # end_voxel=int(sys.argv[5]),
                filterType = filterType,
                transition_cov = transition_cov
            )
        f.close()
    except KeyboardInterrupt:
        # Your code which is executed when CTRL+C is pressed.
        f.close()