# This code should be run in console room computer to display the feedback morphings
from __future__ import print_function, division
#os.chdir("/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtSynth_repo/kp_scratch/expcode")
import os
if 'watts' in os.getcwd():
    main_dir = "/home/watts/Desktop/ntblab/kailong/rtSynth_rt/"
else:
    main_dir="/Users/kailong/Desktop/rtEnv/rtSynth_rt/"
import sys
sys.path.append(main_dir+"expScripts/feedback/")
from psychopy import visual, event, core, logging, gui, data, monitors
from psychopy.hardware.emulator import launchScan, SyncGenerator
from PIL import Image
import string
import fmrisim as sim
import numpy as np
import pandas as pd
import pylink   
from tqdm import tqdm
import time
import re
import argparse
alpha = string.ascii_uppercase

if 'watts' in os.getcwd():
    main_dir = "/home/watts/Desktop/ntblab/kailong/rtSynth_rt/"
else:
    main_dir="/Users/kailong/Desktop/rtEnv/rtSynth_rt/"

# startup parameters
sys.path.append(main_dir)
from rtCommon.cfg_loading import mkdir,cfg_loading

argParser = argparse.ArgumentParser()
argParser.add_argument('--config', '-c', default='sub001.ses2.toml', type=str, help='experiment file (.json or .toml)')
argParser.add_argument('--run', '-r', default='1', type=str, help='current run')
argParser.add_argument('--sess', '-s', default='1', type=str, help='current session')
argParser.add_argument('--server', '-v', default='localhost:7777', type=str, help='current server')
args = argParser.parse_args()

cfg = cfg_loading(args.config)
sub = cfg.subjectName
run = int(args.run)  # 1
sess = int(args.sess)
TR=int(cfg.TR)

cfg.feedback_expScripts_dir = f"{cfg.projectDir}expScripts/feedback/"

if False:
    scanmode = 'Scan'  # 'Scan' or 'Test' or None
    screenmode = True  # fullscr True or False
    monitor_name = "scanner"
else:
    scanmode = 'Test'  # 'Scan' or 'Test' or None
    screenmode = False  # fullscr True or False
    monitor_name = "testMonitor" #"testMonitor"

gui = True if screenmode == False else False
scnWidth, scnHeight = monitors.Monitor(monitor_name).getSizePix()
frameTolerance = 0.001  # how close to onset before 'same' frame
TRduration=2.0


# mywin = visual.Window(
    # size=[1280, 800], fullscr=screenmode, screen=0,
    # winType='pyglet', allowGUI=False, allowStencil=False,
    # monitor=monitor_name, color=[0,0,0], colorSpace='rgb', #color=[0,0,0]
    # blendMode='avg', useFBO=True,
    # units='height')

mywin = visual.Window(
    size=[scnWidth - 100, scnHeight - 100], fullscr=screenmode, screen=1,
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor=monitor_name, color=[0,0,0], colorSpace='rgb', #color=[0,0,0]
    blendMode='avg', useFBO=True,
    units='height')

# similation specific
step=3 #in simulation, how quickly the morph changes ramp up. Note this is only for simulation, has nothing to do with real experiment

# trial_list designing parameters
parameterRange=np.arange(1,11) #for saving time for now. np.arange(1,20) #define the range for possible parameters for preloading images. Preloading images is to make the morphing smooth during feedback
tune=4 # this parameter controls how much to morph (how strong the morphing is) (used in preloading function), tune can range from (1,6.15] when paremeterrange is np.arange(1,20)
TrialNumber=180 # how many trials are required #test trial ,each trial is 14s, 10 trials are 140s.

## - design the trial list: the sequence of the different types of components: 
## - e.g: ITI + waiting for fMRI signal + feedback (receive model output from feedbackReceiver.py)
trial_list = pd.DataFrame(columns=['Trial','time','TR','state','newWobble'])
curTime=0
curTR=0
state=''
trial_list.append({'Trial':None,
                    'time':None,
                    'TR':None,
                    'state':None,
                    'newWobble':None},
                    ignore_index=True)

for currTrial in range(1,1+TrialNumber):

    # ITI
    for i in range(6): # should be 6TR=12s
        trial_list=trial_list.append({'Trial':currTrial,
                                    'time':curTime,
                                    'TR':curTR,
                                    'state':'ITI',
                                    'newWobble':0},
                                    ignore_index=True)
        curTime=curTime+TR
        curTR=curTR+1

    # waiting for metric calculation
    for i in range(3): # should be 3TR=6s
        trial_list=trial_list.append({'Trial':currTrial,
                                    'time':curTime,
                                    'TR':curTR,
                                    'state':'waiting',
                                    'newWobble':0},
                                    ignore_index=True)
        curTime=curTime+TR
        curTR=curTR+1
    
    # feedback trial: try minimize the whobbling
    for i in range(5): #5TR=10s
        trial_list=trial_list.append({'Trial':currTrial,
                                    'time':curTime,
                                    'TR':curTR,
                                    'state':'feedback',
                                    'newWobble':1},
                                    ignore_index=True)
        curTime=curTime+TR
        curTR=curTR+1

# ITI
for i in range(6): # should be 6TR=12s
    trial_list=trial_list.append({'Trial':currTrial,
                                'time':curTime,
                                'TR':curTR,
                                'state':'ITI',
                                'newWobble':0},
                                ignore_index=True)
    curTime=curTime+TR
    curTR=curTR+1

# for currTrial in range(1,1+TrialNumber):
#     for i in range(1): # should be 6TR=12s
#         trial_list=trial_list.append({'Trial':currTrial,
#                                     'time':curTime,
#                                     'TR':curTR,
#                                     'state':'ITI',
#                                     'newWobble':0},
#                                     ignore_index=True)
#         curTime=curTime+TR
#         curTR=curTR+1
#     for i in range(1): # should be 3TR=6s
#         trial_list=trial_list.append({'Trial':currTrial,
#                                     'time':curTime,
#                                     'TR':curTR,
#                                     'state':'waiting',
#                                     'newWobble':0},
#                                     ignore_index=True)
#         curTime=curTime+TR
#         curTR=curTR+1
#     for i in range(5): #5TR=10s
#         trial_list=trial_list.append({'Trial':currTrial,
#                                     'time':curTime,
#                                     'TR':curTR,
#                                     'state':'feedback',
#                                     'newWobble':1},
#                                     ignore_index=True)
#         curTime=curTime+TR
#         curTR=curTR+1
# for i in range(1): # should be 6TR=12s
#     trial_list=trial_list.append({'Trial':currTrial,
#                                 'time':curTime,
#                                 'TR':curTR,
#                                 'state':'ITI',
#                                 'newWobble':0},
#                                 ignore_index=True)
#     curTime=curTime+TR
#     curTR=curTR+1



# parameters = np.arange(1,step*(sum((trial_list['newWobble']==1)*1)),step) #[1,2,3,4,5,6,7,8]

print('total trial number=',TrialNumber)
# print('neighboring morph difference=',tune)
print('preloaded parameter range=',parameterRange)
# print('used parameters=',parameters)



def sample(L,num=10):
    # This functional uniformly sample the list to be num points
    # e.g, if L is 0-99, num is 10, newList would be [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    # e.g, if L is 0-95, num is 10, newList would be [8, 18, 27, 37, 47, 56, 66, 75, 85, 95]
    # e.g, if L is 0-5, num is 10, newList would be [0, 0, 0, 1, 2, 2, 3, 3, 4, 5]
    sampleStep=len(L)/num 
    newList=[]
    for i in range(1,num):
        newList.append(L[int(i*sampleStep-1)])
    newList.append(L[-1])
    return newList


# preload image list for parameter from 1 to 19.
def preloadimages(parameterRange=np.arange(1,20),tune=1):
    '''
    purpose:
        preload images into image object sequences corrresponding too each parameter
        each parameter corresponds to 40 image objects
    steps:
    '''
    tune=tune-1
    start = time.time()
    imageLists={}
    numberOfUpdates=16 # corresponds to 66 updates    
    last_image=''
    for currParameter in tqdm(parameterRange): #49
        images=[]
        print('maximum morph=',round((tune*currParameter*numberOfUpdates+2)/numberOfUpdates+1))
        for axis in ['bedTable', 'benchBed']:
            tmp_images=[]
            for currImg in range(1,int(round(tune*currParameter*numberOfUpdates+2)),int((currParameter*numberOfUpdates+2)/numberOfUpdates)):
                currMorph=100-round(currImg/numberOfUpdates+1) if axis=='benchBed' else round(currImg/numberOfUpdates+1)
                if currMorph<1 or currMorph>99:
                    raise Exception('morphing outside limit')
                curr_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
                if curr_image!=last_image:
                    currImage=visual.ImageStim(win=mywin,
                                                name='image',
                                                image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5), mask=None,
                                                ori=0, pos=(0, 0), size=(0.5, 0.5),
                                                color=[1,1,1], colorSpace='rgb', opacity=1,
                                                flipHoriz=False, flipVert=False,
                                                texRes=128, interpolate=True, depth=-4.0)
                tmp_images.append(currImage)
                last_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
            images=images+sample(tmp_images)
            tmp_images=[]
            for currImg in reversed(range(1,int(round(tune*currParameter*numberOfUpdates+1)),int((currParameter*numberOfUpdates+2)/numberOfUpdates))):
                currMorph=100-round(currImg/numberOfUpdates+1) if axis=='benchBed' else round(currImg/numberOfUpdates+1)
                curr_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
                if curr_image!=last_image:
                    currImage=visual.ImageStim(win=mywin,
                                                name='image',
                                                image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5), mask=None,
                                                ori=0, pos=(0, 0), size=(0.5, 0.5),
                                                color=[1,1,1], colorSpace='rgb', opacity=1,
                                                flipHoriz=False, flipVert=False,
                                                texRes=128, interpolate=True, depth=-4.0)
                tmp_images.append(currImage)
                last_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
            images=images+sample(tmp_images)
        imageLists.update( {currParameter : images} )
    end = time.time()
    print("preload image duration=", end - start)
    return imageLists

imageLists=preloadimages(parameterRange=parameterRange,tune=tune)

# Open data file for eye tracking
# datadir = "./data/feedback/"
datadir = main_dir + f"subjects/{sub}/ses{sess}/feedback/"

maxTR=int(trial_list['TR'].iloc[-1])+6
# Settings for MRI sequence
MR_settings = {'TR': TRduration, 'volumes': maxTR, 'sync': 5, 'skip': 0, 'sound': True} #{'TR': 2.000, 'volumes': maxTR, 'sync': 5, 'skip': 0, 'sound': True}

# check if there is a data directory and if there isn't, make one.
if not os.path.exists('./data'):
    os.mkdir('./data')
if not os.path.exists('./data/feedback/'):
    os.mkdir('./data/feedback/')

# check if data for this subject and run already exist, and raise an error if they do (prevent overwriting)
newfile = datadir+"{}_{}.csv".format(str(sub), str(run))
if os.path.exists(newfile):
    raise Exception(f'{newfile} exists')
# create empty dataframe to accumulate data
data = pd.DataFrame(columns=['Sub', 'Run', 'TR', 'time'])

# Create the fixation dot, and initialize as white fill.
fix = visual.Circle(mywin, units='deg', radius=0.05, pos=(0, 0), fillColor='white',
                    lineColor='black', lineWidth=0.5, opacity=0.5, edges=128)

# start global clock and fMRI pulses (start simulated or wait for real)
print('Starting sub {} in run #{}'.format(sub, run))

vol = launchScan(mywin, MR_settings, simResponses=None, mode=scanmode,
                 esc_key='escape', instr='select Scan or Test, press enter',
                 wait_msg='waiting for scanner...', wait_timeout=300, log=True)

image = visual.ImageStim(
    win=mywin,
    name='image',
    image=cfg.feedback_expScripts_dir + './carchair_exp_feedback/bedChair_1_5.png', mask=None,
    ori=0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)

backgroundImage = visual.ImageStim(
    win=mywin,
    name='image',
    image=cfg.feedback_expScripts_dir+'./carchair_exp_feedback/greyBackground.png', mask=None,
    ori=0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-4.0)

# trialClock is reset in each trial to change image every TR (2s), time for each image is 2/numOfImages
trialClock = core.Clock()

# trialClock.add(10)  # initialize as a big enough number to avoid text being shown at the first time.
TR=list(trial_list['TR'])
states=list(trial_list['state'])
newWobble=list(trial_list['newWobble'])

# parameters=np.round(np.random.uniform(0,10,sum((trial_list['newWobble']==1)*1)))
# parameters = np.arange(1,1+sum((trial_list['newWobble']==1)*1)) #[1,2,3,4,5,6,7,8]
ParameterUpdateDuration=np.diff(np.where(trial_list['newWobble']==1))[0][0]*TRduration
curr_parameter=0
remainImageNumber=[]


# feedbackParameterFileName=main_dir+f"subjects/{sub}/ses{sess}_feedbackParameter/run_{run}.csv"
# # While the running clock is less than the total time, monitor for 5s, which is what the scanner sends for each TR
# _=1
# while not os.path.exists(feedbackParameterFileName):
#     keys = event.getKeys(["5","0"])
#     if '0' in keys: # whenever you want to quite, type 0
#         mywin.close()
#         core.quit()
#     time.sleep(0.01)
#     if _ % 100==0:
#         print(f'waiting {feedbackParameterFileName}')
#     _+=1
# parameters=pd.read_csv(feedbackParameterFileName)
# while np.isnan(parameters['value'].iloc[-1]):
#     keys = event.getKeys(["5","0"])
#     if '0' in keys: # whenever you want to quite, type 0
#         mywin.close()
#         core.quit()
#     time.sleep(0.01)
#     if _ % 100==0:
#         print(f'waiting parameters nan')
#     _+=1
#     parameters=pd.read_csv(feedbackParameterFileName)
from rtCommon.feedbackReceiver import WsFeedbackReceiver
WsFeedbackReceiver.startReceiverThread(args.server,
                                    retryInterval=5,
                                    username="kp578",
                                    password="kp578",
                                    testMode=True)
default_parameter=19
# curr_parameter=len(parameters['value'])-1
while len(TR)>1: #globalClock.getTime() <= (MR_settings['volumes'] * MR_settings['TR']) + 3:
    trialTime = trialClock.getTime()
    keys = event.getKeys(["5","0"])  # check for triggers
    if '0' in keys: # whenever you want to quite, type 0
        mywin.close()
        core.quit()
    if len(keys):
        TR.pop(0)
        states.pop(0)
        newWobble.pop(0)
        print(states[0])
        if states[0] == 'feedback' and newWobble[0]==1:
            # fetch parameter from preprocessing process on Milgram       
            feedbackMsg = WsFeedbackReceiver.msgQueue.get(block=True, timeout=None)     
            runId,trID,value,timestamp=feedbackMsg.get('runId'),\
                feedbackMsg.get('trId'),feedbackMsg.get('value'),feedbackMsg.get('timestamp')

            if value==None:
                parameter = default_parameter
            else:
                parameter = value

            # print('feedbackParameterFileName=',feedbackParameterFileName)
            # parameters=pd.read_csv(feedbackParameterFileName)
            # if curr_parameter>(len(parameters['value'])-1):
                # curr_parameter=curr_parameter-1
            # curr_parameter=(len(parameters['value'])-1)
            # parameter=parameters['value'].iloc[curr_parameter]
            # print('curr_parameter=',curr_parameter)
            # print('parameter=',parameter)
            print(f'TR[0]={TR[0]},trID={trID},parameter={parameter},timestamp={timestamp},runId={runId}')

            # curr_parameter=curr_parameter+1
            # start new clock for current updating duration (the duration in which only a single parameter is used, which can be 1 TR or a few TRs, the begining of the updateDuration is indicated by the table['newWobble'])
            trialClock=core.Clock()
            trialTime=trialClock.getTime()
            # update the image list to be shown based on the fetched parameter
            imagePaths=imageLists[parameter] #list(imageLists[parameter])
            # calculated how long each image should last.
            eachTime=ParameterUpdateDuration/len(imagePaths)
            # update the image
            # image.image=imagePaths[0]
            image.setAutoDraw(False)
            imagePaths[0].setAutoDraw(True)
            # currImage*eachTime is used in the calculation of the start time of next image in the list.
            
            # save when the image is presented and which image is presented.
            data = data.append({'Sub': sub, 
                                'Run': run, 
                                'TR': TR[0],
                                'time': trialTime, 
                                'imageTime':imagePaths[0].image,
                                'eachTime':eachTime},
                               ignore_index=True)
            oldMorphParameter=re.findall(r"_\w+_",imagePaths[0].image)[1]
            # print('curr morph=',oldMorphParameter)
            remainImageNumber.append(0)
            currImage=1
            # # discard the first image since it has been used.
            # imagePaths.pop(0)
    if (states[0] == 'feedback') and (trialTime>currImage*eachTime):
            try: # sometimes the trialTime accidentally surpasses the maximum time, in this case just do nothing, pass
                imagePaths[currImage-1].setAutoDraw(False)
                imagePaths[currImage].setAutoDraw(True)
                # print('currImage=',imagePaths[currImage],end='\n\n')
                remainImageNumber.append(currImage)

                # write the data!
                data = data.append({'Sub': sub, 
                                    'Run': run, 
                                    'TR': TR[0], 
                                    'time': trialTime, 
                                    'imageTime':imagePaths[currImage].image,
                                    'eachTime':eachTime},
                                    ignore_index=True)
                currMorphParameter=re.findall(r"_\w+_",imagePaths[currImage].image)[1]
                if currMorphParameter!=oldMorphParameter:
                    pass
                    # print('curr morph=',currMorphParameter)
                oldMorphParameter=currMorphParameter
                currImage=currImage+1        
            except:
                pass
    elif states[0] == 'ITI':
        backgroundImage.setAutoDraw(True)
        fix.draw()
    elif states[0] == 'waiting':
        backgroundImage.setAutoDraw(False)
        image.setAutoDraw(True)
    # refresh the screen
    mywin.flip()


# write data out!
data.to_csv(newfile)
mywin.close()
core.quit()
