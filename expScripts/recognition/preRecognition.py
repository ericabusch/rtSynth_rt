# preRecognition.py aims to print out the number of TRs for the current recognition run.

'''
input: 
    cfg
    curr_run

output:
    maxTR=xxx

usage example:
    python expScripts/recognition/preRecognition.py -c pilot_sub001.ses1.toml -r 3
'''
import sys,os
if 'watts' in os.getcwd():
    sys.path.append("/home/watts/Desktop/ntblab/kailong/rtSynth_rt/")
elif 'kailong' in os.getcwd():
    sys.path.append("/Users/kailong/Desktop/rtEnv/rtSynth_rt/")
elif 'milgram' in os.getcwd():
    sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')

import argparse
import numpy as np
import pandas as pd
if 'watts' in os.getcwd():
    main_dir = "/home/watts/Desktop/ntblab/kailong/rtcloud_rt/" # main_dir = "/home/watts/Desktop/ntblab/kailong/rtcloud_kp/"
else:
    main_dir="/Users/kailong/Desktop/rtEnv/rtSynth_rt/"

sys.path.append(main_dir)

import rtCommon.utils as utils
from rtCommon.utils import loadConfigFile
argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default='sub001.ses2.toml', type=str, help='experiment file (.json or .toml)')
argParser.add_argument('--run', '-r', default='1', type=str, help='current run')
args = argParser.parse_args()
from rtCommon.cfg_loading import mkdir,cfg_loading
cfg = cfg_loading(args.config)

curr_run=int(args.run)

choose = np.load(f"{cfg.subjects_dir}/{cfg.subjectName}/ses{cfg.session}/recognition/choose.npy")
for curr_run in range(1,1+len(choose)):
    order = f'{cfg.orderFolder}/recognitionOrders_{choose[curr_run - 1]}.csv'
    trial_list = pd.read_csv(order)
    maxTR=trial_list['time'].iloc[-1]/cfg.TR
    print(f"curr_run={curr_run} maxTR={maxTR}")