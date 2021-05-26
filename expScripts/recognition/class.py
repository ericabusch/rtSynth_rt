import os
print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}") 
import sys
import pickle5 as pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import nibabel as nib

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def Class(brain_data,behav_data):
    # metas = bcvar[0]
    # data4d = data[0]
    print([t.shape for t in brain_data])

    accs = []
    for run in range(8):
        testX = brain_data[run]
        testY = behav_data[run]

        trainX=np.zeros((1,1))
        for i in range(8):
            if i !=run:
                trainX=brain_data[i] if trainX.shape==(1,1) else np.concatenate((trainX,brain_data[i]),axis=0)

        trainY = []
        for i in range(8):
            if i != run:
                trainY.extend(behav_data[i])
        clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                 multi_class='multinomial').fit(trainX, trainY)
                
        # Monitor progress by printing accuracy (only useful if you're running a test set)
        acc = clf.score(testX, testY)
        accs.append(acc)
    
    return np.mean(accs)

def getMask(topN, cfg):
    for pn, parc in enumerate(topN):
        _mask = nib.load(f"/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/{subject}/ses1/recognition/mask/GMschaefer_{parc}")
        # schaefer_56.nii.gz
        aff = _mask.affine
        _mask = _mask.get_data()
        _mask = _mask.astype(int)
        # say some things about the mask.
        mask = _mask if pn == 0 else mask + _mask
        mask[mask>0] = 1
    return mask

tmpFile = f"{sys.argv[1]}{int(sys.argv[2])-1}"
print(f"tmpFile={tmpFile}")
[_topN,subject,dataSource,roiloc,N] = load_obj(tmpFile)
[brain_data,behav_data] = load_obj(f"{os.path.dirname(tmpFile)}/{subject}_{dataSource}_{roiloc}_{N}")
_mask=getMask(_topN,subject) ; print('mask dimensions: {}'. format(_mask.shape)) ; print('number of voxels in mask: {}'.format(np.sum(_mask)))
brain_data = [t[:,_mask==1] for t in brain_data]
print("Runs shape", [t.shape for t in brain_data])


sl_result = Class(brain_data, behav_data)
np.save(tmpFile+'_result',sl_result)
print(f"sl_result={sl_result}")