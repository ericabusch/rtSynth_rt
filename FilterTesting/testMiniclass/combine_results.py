# this function takes the output of /gpfs/milgram/project/turk-browne/projects/rtSynth/alex_scratch/realtime/recog_features_parent_kp.py and output _featurematrix which is the input of recog_features copy.py
import os
import nibabel as nib
import numpy as np
from pathlib import Path
import sys
from glob import glob
from tqdm import tqdm
proj_dir = '/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/'
roi_dir = os.path.abspath(os.path.join(proj_dir,'subjects/{}_neurosketch/analysis/firstlevel/rois'))


subject_dir='/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/'
subjects=glob(subject_dir+'*_neurosketch')
subjects=[sub.split('/')[-1].split('_')[0] for sub in subjects if sub.split('/')[-1][0]!='_']
# subjects=['1206161', '1201161', '1206163', '0110171']
subjects_new=[]
for subject in tqdm(subjects):
    try:
        interval = 500
        for roi in ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']:# missing occitemp
            for run in range(1, 7):
                input_dir = f"/gpfs/milgram/scratch60/turk-browne/an633/do_not_filter_first_56s_refit_filter_parallel/{roi}"
                # if not os.path.isdir(f"{input_dir}/{subject}"):
                #     Path(f"{input_dir}/{subject}").mkdir(parents=True, exist_ok=True)
                maskDat = nib.load('{}/{}_func_combined_{}_binarized.nii.gz'.format(roi_dir.format(subject), roi, '12')).get_fdata()
                num_voxels = np.sum(maskDat==1)
                arrays = []
                for start in range(0, num_voxels, interval):
                    old_file_name = f"{input_dir}/{subject}/{roi}_Kalman_filter_{subject}_{run}_featurematrix_voxel_{start}_to_{start+interval}.npy"
                    dat = np.load(old_file_name)
                    arrays.append(dat)


                concat_filtered_voxels = np.hstack(arrays)
                assert concat_filtered_voxels.shape[1] == num_voxels
                # print(run, roi, concat_filtered_voxels.shape)
                new_file_name = f"{input_dir}/{roi}_Kalman_filter_{subject}_{run}_featurematrix.npy"
                np.save(new_file_name, concat_filtered_voxels)
        print(subject)
        subjects_new.append(subject)
    except:
        pass
print(subjects_new)

# ['0110171', '1206161', '0120171', '1206161', '1206163']