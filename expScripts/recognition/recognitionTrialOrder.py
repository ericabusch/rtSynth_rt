# recognition trial order
# This code generate the trial order of recognition run. This is part of the old code in http://127.0.0.1:9206/notebooks/users/kp578/rtSynth/kp_scratch/expcode/recognition%20trial.ipynb

import os
os.chdir('/Volumes/GoogleDrive/My Drive/Turk_Browne_Lab/rtcloud_kp/expScripts/recognition/')

import random,string,pickle
import pandas as pd
import random
from tqdm import tqdm
import numpy as np

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# In total there are 48 trials.
# quarter counterbalance, 3 repetition of each image in each quarter (36 trial)
# prevent back to back repetition 
# generate a new order for each subject and each new recognition run.
TR=2
subj=1
for subj in tqdm(range(1,51)):
    order=[]
    quarter=0

    NumTrial=12 # number Of Trials In A Quater
    NumRep=3 # number Of Repetion Of Single Image In A Quater
    NumImg=4 # number of unique images
    while quarter<4:
        back2backRep_morph=True #indicate that the same morph images are repeated back to back
        # back2backRep_cat=True #indicate that the images in the same cat are repeated back to back 
        # Cat is defined as {'cat1':[A,B,C],'cat2':[D,E,F],'cat3':[G,H,I],'cat4':[J,K,L]}
        while back2backRep_morph: # or back2backRep_cat: # only accept the order when both are false
            _order=np.asarray(random.sample(list(np.arange(NumTrial)),NumTrial))
            for i in range(NumImg):
                _order[np.logical_and(_order>=np.arange(0,NumTrial+1,NumRep)[i],
                                      _order<np.arange(0,NumTrial+1,NumRep)[i+1])]=i
            if 0 not in np.diff(_order):
                back2backRep_morph = False    
            # # prevent same cat back2back repetition
            # _cat=np.array(_order)
            # for i in range(4):
            #     _cat[np.logical_and(_order>=np.arange(0,12+1,3)[i],
            #                           _order<np.arange(0,12+1,3)[i+1])]=i
            # if 0 not in np.diff(_cat):
            #     back2backRep_cat = False    
        #check if this particular sequence already exists in generated orders
        exist=0
        for _ in order:
            if np.all(_order==_):
                exist=1
        if len(order)>0:
            if order[-1][-1]==_order[0]: #avoid tail-head repetion between two quaters
                exist=1
        # if this particular sequence does not already exists in generated orders, store it
        if exist==0:
            order.append(_order)
            quarter=quarter+1
    # convert to a single array
    _=np.array([])
    for a in order:
        _=np.concatenate((_, a), axis=0)
    _order=_
    # convert to letters
    order=[]
    alpha = string.ascii_uppercase
    for i in range(len(_order)):
        order.append(alpha[int(_order[i])])

    # store as csv for records
    class imageProperty():
        def __init__(self):
            self.viewPointOrder=[]
            for i in range(NumImg):
                l=np.arange(3, 40, 3)
                random.shuffle(l)
                self.viewPointOrder.append(list(l))
            save_obj(self.viewPointOrder,'./viewPointOrder/recognitionTrial_'+str(subj))
        def getPath(self,image):
            if image in ['A','B']:
                morphDict={'A':1, 'B':100}
                axis='bedChair'
                button_left='Bed'
                button_right='Chair'
#                 if np.random.binomial(1, 0.5, 1)[0]==1: # randomly switch the button position
#                     button_left='Bed'
#                     button_right='Chair'
#                 else:
#                     button_left='Chair'
#                     button_right='Bed'
            elif image in ['C','D']:
                morphDict={'C':1, 'D':100}
                axis='tableBench'
                button_left='Table'
                button_right='Bench'
                # if np.random.binomial(1, 0.5, 1)[0]==1: # randomly switch the button position
                #     button_left='Table'
                #     button_right='Bench'
                # else:
                #     button_left='Bench'
                #     button_right='Table'
            viewPoint=self.viewPointOrder[alpha.index(image)][0]
            path='./carchair_exp/{}_{}_{}.png'.format(axis,
                                                       morphDict[image],
                                                       viewPoint)
            self.viewPointOrder[alpha.index(image)].pop(0)
            return path,morphDict[image],axis,button_left,button_right,viewPoint
    orders_df = pd.DataFrame(columns=['time','imnum','dur','weight','imcode','path',
                                      'corrAns','axis','button_left','button_right','viewPoint'])
    imagePath=imageProperty()

    cumTime=6
    for currImg in range(len(order)):
        path,morph,axis,button_left,button_right,viewPoint=imagePath.getPath(order[currImg])

        p=np.random.uniform(0,1,1)
        if p<0.4:
            SOA=2*TR
        elif p<0.8:
            SOA=3*TR
        else:
            SOA=4*TR
        orders_df=orders_df.append({'time':cumTime,
                                    'imnum':_order[currImg],
                                    'dur':1.0,
                                    'weight':1,
                                    'imcode':order[currImg],
                                    'imgPath':path,
                                    'corrAns':morph,
                                    'axis':axis,
                                    'button_left':button_left,
                                    'button_right':button_right,
                                    'viewPoint':viewPoint},
                                   ignore_index=True)
        cumTime=cumTime+SOA
    orders_df.to_csv('orders/recognitionOrders_{}.csv'.format(subj))  
