import pandas as pd
import os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def logit(subject,axis,ax,which_subject):
    resp=pd.read_csv('./'+subject, sep='\t', lineterminator='\n',header=None)
    resp=resp.rename(columns={
        0:"workerid",
        1:"this_trial",
        2:"Image",
        3:"ImagePath",
        4:"category",
        5:"imageStart",
        6:"imageEnd",
        7:"response",
        8:"responseTime",
        9:"currentTime",
        })

    tableBench=resp.loc[resp['category'] == axis]
    if len(tableBench)==0:
        return None,None,None,None
    _x=np.asarray(tableBench['Image'])
    X_dict={0:1,1:21,2:41,3:60,4:80,5:100,6:1,7:21,8:41,9:60,10:80,11:100}
    x=[]
    for _ in _x:
        x.append(X_dict[_])
    y=np.asarray(tableBench['response'])
    xy=np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)), axis=0)
    xy=xy[:,xy[0,:].argsort()]
    #xy[1,:]=2-xy[1,:]
    methodFlag="method1"
    if methodFlag=="method1":
        # method 1: use original datapoints for regression
        x=xy[0,:]
        y=xy[1,:]-1
        x=x/100
    else:
        # method 2: use frequency of choices for regression
        prob=[]
        for i in [1,21,41,60,80,100]:
            _prob=np.mean(xy[1,xy[0]==i])
            #print(xy[1,xy[0]==i])
            #print(_prob,end='\n\n')
            prob.append(_prob)
        x=np.asarray([1,21,41,60,80,100], dtype=np.float128)/100
        y=np.asarray(prob)-1
    
    morph1acc=round(np.mean(1-y[x==0.01]),3)
    morph21acc=round(np.mean(1-y[x==0.21]),3)
    morph80acc=round(np.mean(y[x==0.80]),3)
    morph100acc=round(np.mean(y[x==1]),3)
#     print("morph 1 acc=",morph1acc)
#     print("morph 21 acc=",morph21acc)
#     print("morph 80 acc=",morph80acc)
#     print("morph 100 acc=",morph100acc)
    if morph1acc>0.8 and morph100acc>0.8:
        title='✓ '
        exclusion="✓"
    else:
        title='X '
        exclusion="X"
    try: 
        def f(x, k, x0):
            return 1 / (1. + np.exp(-k * (x - x0)))
        # fit and plot the curve
        (k,x0), _ = opt.curve_fit(f, x, y)
        n_plot=100
        x_plot = np.linspace(min(x), max(x), n_plot)
        y_fit = f(x_plot, k,x0)
#         fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        if methodFlag=="method1":
            _=ax.plot(rand_jitter(x), rand_jitter(y), '.')
        else:
            _=ax.plot(x,y, 'o')
        _=ax.plot(x_plot, y_fit, '-')

        if len(tableBench)==72:
            title=title+"{} sub_{}\n k={};x0={}".format(axis,
                                              which_subject, #(subject.split("_")[1]).split(".")[0],
                                              round(k, 2),
                                              round(x0, 2),
                                             )
        else:
            title="X {} sub_{}\n k={};x0={};dataNum={}".format(axis,
                                              which_subject, #(subject.split("_")[1]).split(".")[0],
                                              round(k, 2),
                                              round(x0, 2),
                                              len(tableBench)
                                             )
            exclusion='X'
            
        _=ax.set_title(title,fontdict={'fontsize': 10, 'fontweight': 'medium'})
        
        return morph1acc, morph21acc, morph80acc, morph100acc,k,x0,exclusion,x,y
    except:
        title="X "
        exclusion="X"
        k,x0=None,None
        
        _=ax.plot(rand_jitter(x), rand_jitter(y), '.')
        if len(tableBench)==72:
            title=title+"{} sub_{}".format(axis,
                                           which_subject #(subject.split("_")[1]).split(".")[0],
                                             )
        else:
            title=title+"{} sub_{}\n dataNum={}".format(axis,
                                              which_subject, #(subject.split("_")[1]).split(".")[0],
                                              len(tableBench)
                                             )
        _=ax.set_title(title,fontdict={'fontsize': 10, 'fontweight': 'medium'})
        
        return (morph1acc, morph21acc, morph80acc, morph100acc,k,x0,exclusion,x,y)


def checkVersion(subject):
    resp=pd.read_csv('./'+subject, sep='\t', lineterminator='\n',header=None)
    resp=resp.rename(columns={
        0:"workerid",
        1:"this_trial",
        2:"Image",
        3:"ImagePath",
        4:"category",
        5:"imageStart",
        6:"imageEnd",
        7:"response",
        8:"responseTime",
        9:"currentTime",
        })
    axes={'bedChair':'horizontal', 'benchBed':'vertical', 'chairBench':'diagonal'}
    for axis in ['bedChair', 'benchBed', 'chairBench']:
        data=resp.loc[resp['category'] == axis]
        if len(data)>0:
            return axes[axis]
    
    
def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

os.chdir("/Users/kailong/Desktop/rtEnv/rtSynth_rt/subjects/sub002/ses1/catPer")
subject="catPer_000000sub002_1.txt"
# axis="bedChair" # bedChair tableBench; benchBed chairTable; chairBench bedTable

versionDict={'horizontal':['bedChair', 'tableBench'],
            'vertical':['benchBed', 'chairTable'],
            'diagonal':['chairBench', 'bedTable']}
version=checkVersion(subject)


f, ax = plt.subplots(9,2, figsize=(10, 5*9))
axis=versionDict[version][0]
(morph1acc, morph21acc, morph80acc, morph100acc,k,x0,exclusion,x,y)=logit(subject,axis,ax[0,0],"test")
axis=versionDict[version][1]
(morph1acc, morph21acc, morph80acc, morph100acc,k,x0,exclusion,x,y)=logit(subject,axis,ax[0,1],"test")