# This code is the recognition display code and behavior data(button box) recording code. 
# This code read from the image orders designed by the code expScripts/recognition/recognitionTrialOrder.py 
# and show corresponding images

# The recognition trial design right now is like this: 
# 1000ms stimuli presentation and 900ms 2AFC(alternative forced choice task) and 
# 2000x0.4+4000x0.4+6000x0.2=3600ms SOA(stimulus onset asynchrony)
# There are 48 trials in each order.csv file. So each 



from __future__ import print_function, division
import os

# os.chdir("/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtSynth_repo/kp_scratch/expcode")
from psychopy import visual, event, core, logging, gui, data, monitors
from psychopy.hardware.emulator import launchScan, SyncGenerator
from PIL import Image
import string
import fmrisim as sim
import numpy as np
import pandas as pd
import sys
import os
import pylink

# imcode:
# A: bed
# B: Chair
# C: table
# D: bench

alpha = string.ascii_uppercase

# startup parameters
# maxTR = int((3*0.4+4.5*0.4+6*0.2)*12*4/1.5+60)  # mean 134.4 maximum 192
# sub = sys.argv[1]  # 'test' 'pilot' "pilot_sub002"

from rtCommon.cfg_loading import mkdir,cfg_loading
argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default='pilot_sub001.ses1.toml', type=str, help='experiment file (.json or .toml)')
argParser.add_argument('--run', '-r', default='1', type=str, help='current run')
args = argParser.parse_args()

cfg = cfg_loading(args.config)
sub = cfg.subjectName
run = int(arg.run)  # 1
TR=cfg.TR

scanmode = 'Test'  # 'Scan' or 'Test' or None
screenmode = True  # fullscr True or False
gui = True if screenmode == False else False
monitor_name = "testMonitor" #"scanner" "testMonitor"
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
    main_dir = "/home/watts/Desktop/ntblab/kailong/rtcloud_kp/"
else:
    main_dir="/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtcloud_kp/"

# # Open data file for eye tracking
# datadir = "./data/recognition/"
saveDir="../"

# This sets the order of stimulus presentation for all of the subjects' runs
# If it is the first run, randomly select and save out six orders, otherwise read in that file
# if run == 1:
#     choose = np.random.choice(np.arange(1, 49), 8, replace=False)
#     np.save(f"{main_dir}subjects/{sub}/ses1_recognition/run{run}/{sub}_orders.npy", choose)
# else:
choose = np.load(f"{cfg.subjects_dir}/{cfg.subjectName}/ses{cfg.session}/recognition/choose.npy")

# read the saved order 
order = './orders/recognitionOrders_{}.csv'.format(choose[run - 1])
trial_list = pd.read_csv(order)

maxTR = int(trial_list['time'].iloc[-1] / 2 + 3) 

# Settings for MRI sequence
MR_settings = {'TR': 2.000, 'volumes': maxTR, 'sync': 5, 'skip': 0, 'sound': True}

# check if there is a data directory and if there isn't, make one.
if not os.path.exists('./data'):
    os.mkdir('./data')

# check if data for this subject and run already exist, and raise an error if they do (prevent overwriting)

newfile = f"{main_dir}subjects/{sub}/ses1_recognition/{sub}_{run}.csv"
# log = "./Data/{}_{}.txt".format(str(sub), str(run))
# logfile = open(log, "w")
assert not os.path.isfile(newfile), f"FILE {newfile} ALREADY EXISTS - check subject and run number"

# create empty dataframe to accumulate data
data = pd.DataFrame(columns=['Sub', 'Run', 'TR', 'Onset', 'Item', 'Change', 'CorrResp',
                             'Resp', 'RT', 'Acc', 'image_on', 'button_on', 'button_off'])

# Create the fixation dot, and initialize as white fill.
fix = visual.Circle(mywin, units='deg', radius=0.05, pos=(0, 0), fillColor='white',
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

# Initialize two blank lists -
trials = [] # 'trials' to develop a list of indices pointing to images (for drawing from preloaded images)
changes = [] # 'changes' to develop a list of 1s or 0s to indicate whether there should or should not be a red fixation
count = 0
# For all of the TRs/acquisitions
for i in list(range(maxTR)):
    # If it is a critical TR (i.e. image should be presented), pull the correct list index and red fixation code
    if i in all_TRons:
        im = trial_list['imcode'].iloc[count]
        change = all_changes[count]
        count += 1
    # otherwise, assign these blank or 0 coded values.
    else:
        im = ''
        change = 0
    trials.append(im)
    changes.append(change)

# verify distinct time courses, write out regressor files
stimfunc = []
for letter in alpha[:4]:
    file = open("./data/regressor/{}_{}_{}.txt".format(sub, run, letter), 'w')
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

# start global clock and fMRI pulses (start simulated or wait for real)
print('Starting sub {} in run #{} - list #{}'.format(sub, run, choose[run - 1]))
globalClock = core.Clock()
vol = launchScan(mywin, MR_settings, globalClock=globalClock, simResponses=None, mode=scanmode,
                 esc_key='escape', instr='select Scan or Test, press enter',
                 wait_msg='waiting for scanner...', wait_timeout=300, log=True)

background = visual.ImageStim(
    win=mywin,
    name='background',
    image='./carchair_exp/background.png', mask=None,
    ori=0, pos=(0, 0), size=(1, 1),
    color=[1,1,1], colorSpace='rgb', opacity=0.5,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
button_left_ = visual.TextStim(win=mywin, name='button_left_',
    text='default text',
    font='Arial',
    pos=(-0.27, -0.44), height=0.06, wrapWidth=None, ori=0,
    color='white', colorSpace='rgb', opacity=1,
    languageStyle='LTR',
    depth=-2.0);
button_right_ = visual.TextStim(win=mywin, name='button_right_',
    text='default text',
    font='Arial',
    pos=(0.27, -0.44), height=0.06, wrapWidth=None, ori=0,
    color='white', colorSpace='rgb', opacity=1,
    languageStyle='LTR',
    depth=-3.0);
image = visual.ImageStim(
    win=mywin,
    name='image',
    image='sin', mask=None,
    ori=0, pos=(0, 0.1), size=(0.5, 0.5),
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
# While the running clock is less than the total time, monitor for 5s, which is what the scanner sends for each TR
while globalClock.getTime() <= (MR_settings['volumes'] * MR_settings['TR']) + 3:
    globalTime = globalClock.getTime()
    trialTime = trialClock.getTime()
    keys = event.getKeys(["1", "2", "5", "0"])  # check for triggers / key presses, whenever you want to quite, type 0
    if '0' in keys: # whenever you want to quite, type 0
        mywin.close()
        core.quit()

    if '5' in keys:
                # print(globalClock.getTime())
        trigger_counter += 1  # if there's a trigger, increment the trigger counter
        if len(onsets) != 0:
            # write the data!
            data = data.append({'Sub': sub, 'Run': run, 'TR': trigger_counter - 1, 'Onset': time_list[0],
                                'Item': trials[0], 'Change': changes[0], 'Resp': resp,
                                'RT': resp_time, 'image_on': image_on, 'button_on': button_on,
                                'button_off': button_off},
                               ignore_index=True)
            # pop out all first items, and reset responses, because they correspond to the trial that already happened
            trials.pop(0)  # ['', '', '', '', 'A', '', 'D', '', 'C',...]
            onsets.pop(0)  # ['blank', 'blank', 'blank', 'blank', 'stim', 'blank', 'stim', 'blank', 'stim', 'blank', 'blank', 'stim', 'blank', 'stim', 'blank',...]
            time_list.pop(0)  # [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0,...]
            changes.pop(0)

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
                imgPath = imgPaths[0]
                button_left = button_lefts[0]
                button_right = button_rights[0]
                imgPaths.pop(0)
                button_lefts.pop(0)
                button_rights.pop(0)

                button_left_.setText(button_left)
                button_right_.setText(button_right)
                image.setImage(imgPath)

        sys.stdout.flush()
    if len(keys) > 0:  # if a response is made and its not a 5, document response and RT
        print('keys=',keys)
        if "5" not in keys:
            resp = keys[0]
            resp_time = globalTime - image_on_persist - 1  # when the resp_time is negative, that means the subject press a button before the button appears.
            print('resp_time=', resp_time)
            print('globalTime', globalTime)
            print('button_on', button_on_persist)
            # Print output to the screen so we can monitor performance
            print()
            qual = 'Correctly' if changes[0] == 1 else 'Incorrectly'
            if changes[0] == 1:
                hits += 1
            else:
                falses += 1
            # print('- {} pressed {} after {} s. {} hits, {} FA. -'.format(qual, resp, np.around(resp_time,2), hits, falses))
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

    fix.draw()
    # refresh the screen
    mywin.flip()

# write data out!
data.to_csv(newfile)
mywin.close()
core.quit()
