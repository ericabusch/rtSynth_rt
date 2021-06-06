# This code is the recognition display code and behavior data(button box) recording code. 
# This code read from the image orders designed by the code expScripts/recognition/recognitionTrialOrder.py 
# and show corresponding images

# The recognition trial design is like this: 
# 1000ms stimuli presentation and 900ms 2AFC(alternative forced choice task) and 
# 4000x0.4+6000x0.4+8000x0.2=5600ms SOA(stimulus onset asynchrony)
# There are 48 trials in each order.csv file. So each run is 268.8s=4.48min


from __future__ import print_function, division
import traceback,time
# try:

import sys,os
if 'watts' in os.getcwd():
    sys.path.append("/home/watts/Desktop/ntblab/kailong/rtSynth_rt/")
elif 'kailong' in os.getcwd():
    sys.path.append("/Users/kailong/Desktop/rtEnv/rtSynth_rt/")
elif 'milgram' in os.getcwd():
    sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')

import os
from psychopy import visual, event, core, logging, gui, data, monitors
from psychopy.hardware.emulator import launchScan, SyncGenerator
from PIL import Image
import string
import numpy as np
import pandas as pd
import pylink
import argparse
import rtCommon.fmrisim as sim
from rtCommon.cfg_loading import mkdir,cfg_loading

# imcode:
# A: bed
# B: Chair
# C: table
# D: bench
alpha = string.ascii_uppercase


argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default='sub001.ses1.toml', type=str, help='experiment file (.json or .toml)')
argParser.add_argument('--run', '-r', default='1', type=str, help='current run')
argParser.add_argument('--trying', default=False, action='store_true',
                        help='Use unsecure non-encrypted connection')
args = argParser.parse_args()

cfg = cfg_loading(args.config)
sub = cfg.subjectName
run = int(args.run)  # 1
TR=cfg.TR

if args.trying:
    scanmode = 'Test'  # 'Scan' or 'Test' or None
    screenmode = False  # fullscr True or False
    monitor_name = "testMonitor" #"scanner" "testMonitor"
else:
    scanmode = 'Scan'  # 'Scan' or 'Test' or None
    screenmode = True  # fullscr True or False
    monitor_name = "scanner" #"scanner" "testMonitor"

gui = True if screenmode == False else False
scnWidth, scnHeight = monitors.Monitor(monitor_name).getSizePix()
frameTolerance = 0.001  # how close to onset before 'same' frame

# # create window on which all experimental stimuli will be drawn.
# mywin = visual.Window([scnWidth - 10, scnHeight - 10], color=(0, 0, 0), screen=1, units="pix",
#                       monitor=monitor_name, fullscr=screenmode, waitBlanking=False, allowGUI=gui)

# Setup the Window
# mywin = visual.Window(
#     size=[1280, 800], fullscr=screenmode, screen=0,
#     winType='pyglet', allowGUI=False, allowStencil=False,
#     monitor=monitor_name, color=[0,0,0], colorSpace='rgb',
#     blendMode='avg', useFBO=True,
#     units='height')
mywin = visual.Window(
    size=[scnWidth - 100, scnHeight - 100], fullscr=screenmode, screen=1,
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor=monitor_name, color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True,
    units='height')

if 'watts' in os.getcwd():
    main_dir = "/home/watts/Desktop/ntblab/kailong/rtSynth_rt/" # main_dir = "/home/watts/Desktop/ntblab/kailong/rtSynth_rt/"
else:
    main_dir="/Users/kailong/Desktop/rtEnv/rtSynth_rt/"

# This sets the order of stimulus presentation for all of the subjects' runs
# If it is the first run, randomly select and save out six orders, otherwise read in that file
# if run == 1:
#     choose = np.random.choice(np.arange(1, 49), 8, replace=False)
#     np.save(f"{main_dir}subjects/{sub}/ses1_recognition/run{run}/{sub}_orders.npy", choose)
# else:
choose = np.load(f"{cfg.subjects_dir}/{cfg.subjectName}/ses{cfg.session}/recognition/choose.npy")

# read the saved order 
if args.trying:
    order = f'{cfg.recognition_expScripts_dir}/orders/recognitionOrders_trying.csv'
else:
    if len(choose)>(run-1):
        order = f'{cfg.recognition_expScripts_dir}/orders/recognitionOrders_{choose[run - 1]}.csv'
    else:
        choose=list(choose)
        choose.append(int(np.random.choice(np.arange(1, 49), 1)))
        choose=np.asarray(choose)
        print(f"-------------------------------------------- \n appending choose file!!!!! current run = {run}; \n new choose={choose}")
        np.save(f"{cfg.subjects_dir}/{cfg.subjectName}/ses{cfg.session}/recognition/choose.npy",choose)
trial_list = pd.read_csv(order)

# 丢弃最开始的几个TR。总共需要290s/2=145TR,再加上我需要的24s，那么recognition一共是145+12=157 TR
# Jeff: Maybe avoid having an onset for at least the first 4 TRs
countDown = 3 #决定丢弃每个run开始的6个TR，实际上我的order里面还有6s的固定值，因此从scan开始到第一张图片出现过去了6+6=12s
endCountDown = 7 #每个run结束有7个TR的blank，实际上由于之前吃掉了9个TR，现在7个TR的显示只有6个TR

maxTR = int(trial_list['time'].iloc[-1] / 2 + countDown + endCountDown) 
print(f"maxTR={maxTR}")

# Settings for MRI sequence
MR_settings = {'TR': np.float(cfg.TR), 'volumes': maxTR, 'sync': 5, 'skip': 0, 'sound': True}

# check if there is a data directory and if there isn't, make one.
mkdir('./data')

# check if data for this subject and run already exist, and raise an error if they do (prevent overwriting)

newfile = f"{main_dir}subjects/{sub}/ses{cfg.session}/recognition/{sub}_{run}.csv"
assert not os.path.isfile(newfile), f"FILE {newfile} ALREADY EXISTS - check subject and run number"

# create empty dataframe to accumulate data
data = pd.DataFrame(columns=['Sub', 'Run', 'TR', 'Onset', 'Item', 'Change', 'CorrResp',
                            'Resp', 'RT', 'Acc', 'image_on', 'button_on', 'button_off'])

# Create the fixation dot, and initialize as white fill.
fix = visual.Circle(mywin, units='deg', radius=0.05, pos=(0, 0+5), fillColor='white',
                    lineColor='black', lineWidth=0.5, opacity=0.5, edges=128)

# Grab all onsets from the 'order' file. They are in seconds, so convert to TR units
all_onsets = np.array(trial_list['time'])
all_TRons = np.array((all_onsets / TR).astype(int))
# Generate an array of matching size to control tracking task (press 1 if the fixation turns black)
all_changes = np.zeros(all_TRons.shape)
# # Select a random subset of these (but not the first 3) to be 'red' trials, and assign these to the array
# randinds = np.random.choice(np.arange(3, all_TRons.shape[0]), int(all_TRons.shape[0]/10), replace = False)
# all_changes[randinds] = 1

# This section moves away from controlling 80 onsets, to controlling maxTR TRs/acquisitions
# list comprehension to fill in whether a stimulus should be shown or not
onsets = ['blank' if i not in all_TRons else 'stim' for i in list(range(maxTR))]
# Create a list containing the corresponding temporal onsets in seconds
time_list = list(np.arange(0, (TR * maxTR), TR))
imgPaths = list(trial_list['imgPath'])
button_lefts = list(trial_list['button_left'])
button_rights = list(trial_list['button_right'])
ims = list(trial_list['imcode'])
# Trial_ID = list(trial_list['Unnamed: 0'])
# Trial_ID.append(Trial_ID[-1]+1)
imcodeDict={
'A': 'bed',
'B': 'chair',
'C': 'table',
'D': 'bench'}
correctResponseDict={
'A': 1,
'B': 2,
'C': 1,
'D': 2}
# Initialize two blank lists -
trials = [] # 'trials' to develop a list of indices pointing to images (for drawing from preloaded images)
changes = [] # 'changes' to develop a list of 1s or 0s to indicate whether there should or should not be a red fixation
Trial_ID = []
count = 0
# For all of the TRs/acquisitions
for i in list(range(maxTR)):
    # If it is a critical TR (i.e. image should be presented), pull the correct list index and red fixation code
    if i in all_TRons:
        im = trial_list['imcode'].iloc[count]
        curr_trial_ID =  trial_list['Unnamed: 0'].iloc[count]
        change = all_changes[count]
        count += 1
    # otherwise, assign these blank or 0 coded values.
    else:
        curr_trial_ID = ''
        im = ''
        change = 0
    trials.append(im)
    Trial_ID.append(curr_trial_ID)
    changes.append(change)

# verify distinct time courses, write out regressor files
stimfunc = []
for letter in alpha[:4]:
    mkdir(f"{cfg.recognition_dir}regressor/")
    file = open(f"{cfg.recognition_dir}regressor/{run}_{letter}.txt", 'w')
    thiscode = trial_list[trial_list['imcode'] == letter]
    blanks = np.zeros((maxTR, 1))
    TRons = np.array((thiscode['time'] / TR).astype(int))
    blanks[TRons] = 1
    stimfunc = blanks if len(stimfunc) == 0 else np.hstack((stimfunc, blanks))
    for row in TRons:
        file.write(str(row * TR) + f" {str(TR)} 1.00\n")
    file.close()

timeCourse = sim.convolve_hrf(stimfunc, TR, temporal_resolution=1 / TR)
corrs = np.corrcoef(np.transpose(timeCourse))
inds = np.triu_indices(4, k=1)
relcorrs = corrs[inds]
print('time course intercorrelations, min {}, median {} max {}'.format(np.around(np.amin(relcorrs), 2),
                                                                    np.around(np.median(relcorrs), 2),
                                                                    np.around(np.amax(relcorrs), 2)))

# startup terms
trigger_counter = 0
resp = ""
resp_time = ""
image_on = ""
button_on = ""
button_off = ""




background = visual.ImageStim(
    win=mywin,
    name='background',
    image=f'{cfg.recognition_expScripts_dir}carchair_exp/background.png', mask=None,
    ori=0, pos=(0, 0+0.15), size=(1*0.6, 1*0.6),
    color=[1,1,1], colorSpace='rgb', opacity=0.5,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
button_left_ = visual.TextStim(win=mywin, name='button_left_',
    text='',
    font='Arial',
    pos=(-0.27*0.6, -0.44*0.6+0.15), height=0.06*0.6, wrapWidth=None, ori=0,
    color='white', colorSpace='rgb', opacity=1,
    languageStyle='LTR',
    depth=-2.0)
button_right_ = visual.TextStim(win=mywin, name='button_right_',
    text='',
    font='Arial',
    pos=(0.27*0.6, -0.44*0.6+0.15), height=0.06*0.6, wrapWidth=None, ori=0,
    color='white', colorSpace='rgb', opacity=1,
    languageStyle='LTR',
    depth=-3.0)
image = visual.ImageStim(
    win=mywin,
    name='image',
    image='sin', mask=None,
    ori=0, pos=(0, 0.1+0.15), size=(0.5*0.6, 0.5*0.6),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)

# main loop, initialize 0 hits and 0 false alarms for tracking task
hits = 0
falses = 0
# curr_image = preloadims[0]

# flags indicating the timepoint for 1s and 1.9s for each trial, turn to 1 at the beggining of image presentation
# and becomes 0 when 1s and 1.9s is reached respectively.
time1s = 0
time19s = 0

trialClock = core.Clock()
trialClock.add(10)  # initialize as a big enough number to avoid text being shown at the first time.
image_status = 0
button_on_persist = 0
image_on_persist = 0

message = visual.TextStim(mywin, text=f'Waiting...',pos=(0, 0), depth=-5.0, height=32,units='pix')
message.setAutoDraw(False)
def display(countDown,message): #endMorphing can be [1,5,9,13]
    message.setAutoDraw(False)
    message = visual.TextStim(mywin, text=f'Waiting for {2*countDown} s',pos=(0, 0), depth=-5.0, height=32,units='pix')
    message.setAutoDraw(True)
    return message
responseNumber = 0
TrialNumber = 0
secondCounter=1

# start global clock and fMRI pulses (start simulated or wait for real)
print('Starting sub {} in run #{} - list #{}'.format(sub, run, choose[run - 1]))
globalClock = core.Clock()
vol = launchScan(mywin, MR_settings, globalClock=globalClock, simResponses=None, mode=scanmode,
                esc_key='escape', instr='select Scan or Test, press enter',
                wait_msg='waiting for scanner...', wait_timeout=300, log=True)

# While the running clock is less than the total time, monitor for 5s, which is what the scanner sends for each TR
while globalClock.getTime() <= (MR_settings['volumes'] * MR_settings['TR']): # 其中(MR_settings['volumes'] * MR_settings['TR']) 是纯粹的trial需要的时间以及加上前面放弃的countDown=6也就是12s，再加上最后的12s空白的和。
    globalTime = globalClock.getTime()
    if globalTime > secondCounter:
        print(f"{secondCounter} passed")
        secondCounter+=1

    trialTime = trialClock.getTime()
    keys = event.getKeys(["1", "2", "5", "0"])  # check for triggers / key presses, whenever you want to quite, type 0
    if '0' in keys: # whenever you want to quit, type 0
        mywin.close()
        core.quit()

    if '5' in keys:
                # print(globalClock.getTime())
        trigger_counter += 1  # if there's a trigger, increment the trigger counter。当前是第几个TR？
        if countDown > 0:
            message.setAutoDraw(False)
            message = display(countDown,message)
            countDown-=1
        elif len(onsets) != 0:
            message.setAutoDraw(False)
            # write the data!
            print(f"running trial {Trial_ID[0]}")
            data = data.append({'Trial_ID':Trial_ID[0],'Sub': sub, 'Run': run, 'TR': trigger_counter - 1, 'Onset': time_list[0],
                                'Item': trials[0], 'Change': changes[0], 'Resp': resp,
                                'RT': resp_time, 'image_on': image_on, 'button_on': button_on,
                                'button_off': button_off},
                                ignore_index=True)
            data.to_csv(newfile) 
            # pop out all first items, and reset responses, because they correspond to the trial that already happened
            trials.pop(0)  # ['', '', '', '', 'A', '', 'D', '', 'C',...]
            onsets.pop(0)  # ['blank', 'blank', 'blank', 'blank', 'stim', 'blank', 'stim', 'blank', 'stim', 'blank', 'blank', 'stim', 'blank', 'stim', 'blank',...]
            time_list.pop(0)  # [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0,...]
            changes.pop(0)
            Trial_ID.pop(0)
            
            image_on = ""
            button_on = ""
            button_off = ""
            resp = ""
            resp_time = ""
            print(trigger_counter, end=", ")
            if len(onsets)==0:
                break
            if onsets[0] == 'stim':
                trialClock = core.Clock()
                trialTime = trialClock.getTime()
                print('image ON', globalTime)
                image_on = globalTime
                image_on_persist = globalTime  # the purpose of this variable is to make the latest image_on global time available for response time caculation
                time1s = 1
                time19s = 1
                imgPath = f"{cfg.recognition_expScripts_dir}{imgPaths[0]}"
                button_left = button_lefts[0]
                button_right = button_rights[0]
                imgPaths.pop(0)
                img=ims[0]
                button_lefts.pop(0)
                button_rights.pop(0)
                ims.pop(0)
                

                button_left_.setText(button_left)
                button_right_.setText(button_right)
                image.setImage(imgPath)

        sys.stdout.flush()
    if len(keys) > 0:  # if a response is made and its not a 5, document response and RT
        print('keys=',keys)
        if "5" not in keys:
            responseNumber+=1
            resp = keys[0]
            resp_time = globalTime - image_on_persist - 1  # when the resp_time is negative, that means the subject press a button before the button appears.
            print('resp_time=', resp_time)
            print('globalTime', globalTime)
            print('button_on', button_on_persist)
            # Print output to the screen so we can monitor performance
            print()

            
            qual = 'Correctly' if correctResponseDict[img]==int(resp) else 'Incorrectly'
            if correctResponseDict[img]==int(resp):
                hits += 1
            else:
                falses += 1
            print('- {} pressed {} after {} s. {} right, {} wrong. -'.format(qual, resp, np.around(resp_time,2), hits, falses))
            event.clearEvents()

    if len(onsets) != 0:  # if there are still trials remaining, draw the trial
        nindex = alpha.index(trials[0])  # turn image letter code into number
        if onsets[0] == 'stim':  # if there should be a stimulus presented, pull it from preloaded list and draw it.
            image.setAutoDraw(True)
            image_status = 1
            background.setAutoDraw(True)

    if image_status == 1 and trialTime >= 1 - frameTolerance:
        # background.setAutoDraw(False)
        image.setAutoDraw(False)
        image_status = 0

    if button_left_.status <= 0 and trialTime >= 1 - frameTolerance:
        # background.setAutoDraw(True)
        button_left_.setAutoDraw(True)
        button_right_.setAutoDraw(True)
        if time1s == 1:
            print('text ON', globalTime, trialTime)
            time1s = 0
            button_on = globalTime  # data.at[lastestImageIDinData,'button_on']
            button_on_persist = globalTime  # the purpose of this variable is to make the latest button_on global time available for response time caculation

    if button_left_.status == 1:
        # is it time to stop? (based on global clock, using actual start)
        if trialTime >= 1.9 - frameTolerance:
            # background.setAutoDraw(False)
            button_left_.setAutoDraw(False)
            button_right_.setAutoDraw(False)
            background.setAutoDraw(False)
            if time19s == 1:
                print('text OFF', globalTime, trialTime)
                button_off = globalTime  # data.at[lastestImageIDinData,'button_off']=globalTime
                time19s = 0
                TrialNumber+=1
                print("================================================")
                print(f"missed {TrialNumber-responseNumber}")

    fix.draw()
    # refresh the screen
    mywin.flip()

# write data out!
data.to_csv(newfile)
mywin.close()
core.quit()
# except Exception as e:
#     print(f"error {e}")
#     with open(f'./logs/log_{time.time()}.txt', 'a') as f:
#         f.write(str(e))
#         f.write(traceback.format_exc())
    