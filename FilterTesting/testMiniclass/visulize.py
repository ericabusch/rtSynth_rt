
import pandas as pd
import numpy as np
# load and plot data
accuracyContainer=[]
testEvidence=[]
for include in [0.1,0.3,0.6,0.9,1]:
    for roi in ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']:
        for filterType in ['noFilter','highPassRealTime','highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']:
            model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{np.float(include)}/{roi}/{filterType}/'
            accuracyContainer.append(pd.read_csv(f"{model_folder}accuracy.csv"))
            testEvidence.append(pd.read_csv(f'{model_folder}testEvidence.csv'))
accuracyContainer=pd.concat(accuracyContainer, ignore_index=True)
testEvidence=pd.concat(testEvidence, ignore_index=True)
print('testEvidence=',testEvidence)
