# This code should be run in console room computer to display the feedback morphings
from __future__ import print_function, division
import os
if 'watts' in os.getcwd():
    main_dir = "/home/watts/Desktop/ntblab/kailong/rtSynth_rt/"
else:
    main_dir="/Users/kailong/Desktop/rtEnv/rtSynth_rt/"
import sys
sys.path.append(main_dir)
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
import logging
import threading
import argparse
alpha = string.ascii_uppercase
from rtCommon.subjectInterface import SubjectInterface
from rtCommon.wsRemoteService import WsRemoteService, parseConnectionArgs
from rtCommon.utils import installLoggers
from rtCommon.cfg_loading import mkdir,cfg_loading



class SubjectService:
    def __init__(self, args, webSocketChannelName='wsSubject'):
        """
        Uses the WsRemoteService framework to parse connection-related args and establish
        a connection to a remote projectServer. Instantiates a local version of
        SubjectInterface to handle client requests coming from the projectServer connection.
        Args:
            args: Argparse args related to connecting to the remote server. These include
                "-s <server>", "-u <username>", "-p <password>", "--test",
                "-i <retry-connection-interval>"
            webSocketChannelName: The websocket url extension used to connecy and communicate
                to the remote projectServer, 'wsSubject' will connect to 'ws://server:port/wsSubject'
        """
        self.subjectInterface = SubjectInterface(subjectRemote=False)
        self.wsRemoteService = WsRemoteService(args, webSocketChannelName)
        self.wsRemoteService.addHandlerClass(SubjectInterface, self.subjectInterface)

    def runDetached(self):
        """Starts the receiver in it's own thread."""
        self.recvThread = threading.Thread(name='recvThread',
                                           target=self.wsRemoteService.runForever)
        self.recvThread.setDaemon(True)
        self.recvThread.start()


argParser = argparse.ArgumentParser()

argParser.add_argument('-c', '--config', action="store", dest="config", default='sub001.ses2.toml', type=str, help='experiment file (.json or .toml)')
argParser.add_argument('-r', '--run', action="store", dest="run", default='1', type=str, help='current run')
# argParser.add_argument('-e', '--sess', action="store", dest="sess", default='1', type=str, help='current session')
argParser.add_argument('-s', action="store", dest="server", default="localhost:7777",
                    help="Server Address with Port [server:port]")

argParser.add_argument('-i', action="store", dest="interval", type=int, default=5,
                    help="Retry connection interval (seconds)")
argParser.add_argument('-u', '--username', action="store", dest="username", default='kp578',
                    help="rtcloud website username")
argParser.add_argument('-p', '--password', action="store", dest="password", default='kp578',
                    help="rtcloud website password")
argParser.add_argument('--test', default=False, action='store_true',
                    help='Use unsecure non-encrypted connection')
argParser.add_argument('--trying', default=False, action='store_true',
                    help='Use unsecure non-encrypted connection')
args = argParser.parse_args("")

args.trying=True
if args.trying:
    scanmode = 'Test'  # 'Scan' or 'Test' or None
    screenmode = False  # fullscr True or False
    monitor_name = "testMonitor" #"testMonitor"
    prange=20
else:
    scanmode = 'Scan'  # 'Scan' or 'Test' or None
    screenmode = True  # fullscr True or False
    monitor_name = "scanner"
    prange=20

if not re.match(r'.*:\d+', args.server):
    print("Error: Expecting server address in the form <servername:port>")
    argParser.print_help()
    sys.exit()

# Check if the ssl certificate is valid for this server address
from rtCommon.projectUtils import login, certFile, checkSSLCertAltName, makeSSLCertFile
addr, _ = args.server.split(':')
if checkSSLCertAltName(certFile, addr) is False:
    # Addr not listed in sslCert, recreate ssl Cert
    makeSSLCertFile(addr)


cfg = cfg_loading(args.config)
sub = cfg.subjectName
run = int(args.run)  # 1
sess = int(cfg.session)

cfg.feedback_expScripts_dir = f"{cfg.projectDir}expScripts/feedback/"

gui = True if screenmode == False else False
scnWidth, scnHeight = monitors.Monitor(monitor_name).getSizePix()
frameTolerance = 0.001  # how close to onset before 'same' frame
TRduration=int(cfg.TR)

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
parameterRange=np.arange(1,prange) #for saving time for now. np.arange(1,20) #define the range for possible parameters for preloading images. Preloading images is to make the morphing smooth during feedback
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
        curTime=curTime+TRduration
        curTR=curTR+1

    # waiting for metric calculation
    for i in range(3): # should be 3TR=6s
        trial_list=trial_list.append({'Trial':currTrial,
                                    'time':curTime,
                                    'TR':curTR,
                                    'state':'waiting',
                                    'newWobble':0},
                                    ignore_index=True)
        curTime=curTime+TRduration
        curTR=curTR+1
    
    # feedback trial: try minimize the whobbling
    for i in range(5): #5TR=10s
        trial_list=trial_list.append({'Trial':currTrial,
                                    'time':curTime,
                                    'TR':curTR,
                                    'state':'feedback',
                                    'newWobble':1},
                                    ignore_index=True)
        curTime=curTime+TRduration
        curTR=curTR+1

# end ITI
for i in range(6): # should be 6TR=12s
    trial_list=trial_list.append({'Trial':currTrial,
                                'time':curTime,
                                'TR':curTR,
                                'state':'ITI',
                                'newWobble':0},
                                ignore_index=True)
    curTime=curTime+TRduration
    curTR=curTR+1



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
# def preloadimages(parameterRange=np.arange(1,20),tune=1):
#     '''
#     purpose:
#         preload images into image object sequences corrresponding too each parameter
#         each parameter corresponds to 40 image objects
#     steps:

#         重新说明一下image 的命名方式：
#             [benchBed/bedChair/bedTable/]   _   [1-99]              _   [5-39]              .png
#             [AD/AB/AC]                      _   [morphing degree]   _   [angle to watch]    .png  

#             {"A": "bed", "B": "Chair", "C": "table", "D": "bench"}


#     '''
#     tune=tune-1 #当前tune=4
#     #当前tune=3
#     start = time.time()
#     imageLists={}
#     numberOfUpdates=16 # corresponds to 66 updates    A - 15 + C1 + 15 + 1 + 15 + 1 
#                         #  A 10 C 10 A 10 D 10 A
#     last_image=''
#     for currParameter in tqdm(parameterRange):
#         images=[]
#         print('maximum morph=',round((tune*currParameter*numberOfUpdates+2)/numberOfUpdates+1))
#         for axis in ['bedTable', 'benchBed']:
#             tmp_images=[]
#             for currImg in range(1, int(round(tune*currParameter*numberOfUpdates+2)), int((currParameter*numberOfUpdates+2)/numberOfUpdates)):
#                 currMorph=100-round(currImg/numberOfUpdates+1) if axis=='benchBed' else round(currImg/numberOfUpdates+1)
                
#                 # 检查morph的范围是在1-99之间
#                 if currMorph<1 or currMorph>99:
#                     raise Exception('morphing outside limit')

#                 curr_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
#                 if curr_image!=last_image:
#                     currImage = cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
#                 tmp_images.append(currImage)
#                 last_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
#             # images=images+sample(tmp_images) # 将原本比较长的tmp_images选择其中的10个来作为新的短的tmp_images
#             images=images+tmp_images

#             tmp_images=[]
#             for currImg in reversed(range(1,int(round(tune*currParameter*numberOfUpdates+1)),int((currParameter*numberOfUpdates+2)/numberOfUpdates))):
#                 currMorph=100-round(currImg/numberOfUpdates+1) if axis=='benchBed' else round(currImg/numberOfUpdates+1)
#                 curr_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
#                 if curr_image!=last_image:
#                     currImage = cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
#                 tmp_images.append(currImage)
#                 last_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
#             # images=images+sample(tmp_images)
#             images=images+tmp_images
#         imageLists[currParameter]=images

#     end = time.time()
#     print("preload image duration=", end - start)
#     return imageLists

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
                    # currImage=visual.ImageStim(win=mywin,
                    #                             name='image',
                    #                             image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5), mask=None,
                    #                             ori=0, pos=(0, 0), size=(0.5, 0.5),
                    #                             color=[1,1,1], colorSpace='rgb', opacity=1,
                    #                             flipHoriz=False, flipVert=False,
                    #                             texRes=128, interpolate=True, depth=-4.0)
                    currImage = cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
                tmp_images.append(currImage)
                last_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
            images=images+sample(tmp_images)
            tmp_images=[]
            for currImg in reversed(range(1,int(round(tune*currParameter*numberOfUpdates+1)),int((currParameter*numberOfUpdates+2)/numberOfUpdates))):
                currMorph=100-round(currImg/numberOfUpdates+1) if axis=='benchBed' else round(currImg/numberOfUpdates+1)
                curr_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
                if curr_image!=last_image:
                    # currImage=visual.ImageStim(win=mywin,
                    #                             name='image',
                    #                             image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5), mask=None,
                    #                             ori=0, pos=(0, 0), size=(0.5, 0.5),
                    #                             color=[1,1,1], colorSpace='rgb', opacity=1,
                    #                             flipHoriz=False, flipVert=False,
                    #                             texRes=128, interpolate=True, depth=-4.0)
                    currImage = cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
                tmp_images.append(currImage)
                last_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
            images=images+sample(tmp_images)
        imageLists.update( {currParameter : images} )
    end = time.time()
    print("preload image duration=", end - start)
    return imageLists

_=time.time()
imageLists=preloadimages(parameterRange=np.arange(1,33),tune=tune)

print(f"len(imageLists)={len(imageLists)}")
print(f"len(imageLists[1])={len(imageLists[1])}") # 97 97 = 194


'''图片转化为视频'''
def pic2vid(imgList,save2=''): 
    import cv2
    img_array = []
    for filename in imgList:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    fps = 20
    out = cv2.VideoWriter(f'/Users/kailong/Downloads/{save2}.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size) 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

pic2vid(imageLists[32],save2="32")



"""如何理解图片顺序的设计"""
    tune=4

    parameterRange=np.arange(1,20)
    tune=1
    tune=tune-1 #当前tune=4
    #当前tune=3
    start = time.time()
    imageLists={}
    numberOfUpdates=16 # corresponds to 66 updates    A - 15 + C1 + 15 + 1 + 15 + 1 
                        #  A 10 C 10 A 10 D 10 A
    last_image=''
    for currParameter in tqdm(parameterRange):
        images=[]
        print('maximum morph=',round((tune*currParameter*numberOfUpdates+2)/numberOfUpdates+1))
        for axis in ['bedTable', 'benchBed']:
            tmp_images=[]
            for currImg in range(1, int(round(tune*currParameter*numberOfUpdates+2)), int((currParameter*numberOfUpdates+2)/numberOfUpdates)):
                # 计算当前的morph程度
                currMorph=100-round(currImg/numberOfUpdates+1) if axis=='benchBed' else round(currImg/numberOfUpdates+1)

                # 避免morph不在1-99
                if currMorph<1 or currMorph>99:
                    raise Exception('morphing outside limit')
                # 该morph对应的图片的文件
                currImage = cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
                tmp_images.append(currImage)
                
            images=images+sample(tmp_images)
            tmp_images=[]
            for currImg in reversed(range(1,int(round(tune*currParameter*numberOfUpdates+1)),int((currParameter*numberOfUpdates+2)/numberOfUpdates))):
                currMorph=100-round(currImg/numberOfUpdates+1) if axis=='benchBed' else round(currImg/numberOfUpdates+1)
                curr_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
                if curr_image!=last_image:
                    currImage = image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
                tmp_images.append(currImage)
                last_image=cfg.feedback_expScripts_dir+'carchair_exp_feedback/{}_{}_{}.png'.format(axis,currMorph,5)
            images=images+sample(tmp_images)
        imageLists.update( {currParameter : images} )
    end = time.time()
    print("preload image duration=", end - start)