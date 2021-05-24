
# design 13  and random button
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
        10:"ButtonLeft",
        11:"ButtonRight"
        })

    tableBench=resp.loc[resp['category'] == axis]
    if len(tableBench)==0:
        return None,None,None,None
    _x=np.asarray(tableBench['Image'])
    # X_dict={0:18, 1:26, 2:34, 3:42, 4:50, 5:58, 6:66, 7:74, 8:82, 9:18, 10:26, 11:34, 12:42, 13:50, 14:58, 15:66, 16:74, 17:82}
    # X_dict={0:18, 1:26, 2:34, 3:42, 4:50, 5:58, 6:66, 7:74, 8:82, 9:18, 10:26, 11:34, 12:42, 13:50, 14:58, 15:66, 16:74, 17:82}
    X_dict={0:18,1:26,2:34,3:38,4:42,5:46,6:50,7:54,8:58,9:62,10:66,11:74,12:82,13:18,14:26,15:34,16:38,17:42,18:46,19:50,20:54,21:58,22:62,23:66,24:74,25:82}
    x=[]
    for _ in _x:
        x.append(X_dict[_])
    y=np.asarray(tableBench['response'])
    # according to whether bottonLeft is Bed or Table, resave the response
    y_=[]
    if 'bed' in axis:
        button_good=list(tableBench['ButtonLeft']=="Bed")
    else:
        button_good=list(tableBench['ButtonLeft']=="Table")
    for i,j in enumerate(button_good):
        if j:
            y_.append(y[i])
        else:
            if y[i]==1:
                y_.append(2)
            else:
                y_.append(1)
    y=np.asarray(y_)

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
        pass
        # prob=[]
        # for i in [18, 26, 34, 42, 50, 58, 66, 74, 82]:
        #     _prob=np.mean(xy[1,xy[0]==i])
        #     #print(xy[1,xy[0]==i])
        #     #print(_prob,end='\n\n')
        #     prob.append(_prob)
        # x=np.asarray([18, 26, 34, 42, 50, 58, 66, 74, 82], dtype=np.float128)/100
        # y=np.asarray(prob)-1
    
    morph1acc=round(np.mean(1-y[x==0.01]),3)
    morph21acc=round(np.mean(1-y[x==0.21]),3)
    morph80acc=round(np.mean(y[x==0.80]),3)
    morph100acc=round(np.mean(y[x==1]),3)
    # print("morph 1 acc=",morph1acc)
    # print("morph 21 acc=",morph21acc)
    # print("morph 80 acc=",morph80acc)
    # print("morph 100 acc=",morph100acc)
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
        # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
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
    stdev = .005*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

# os.chdir("/Users/kailong/Desktop/rtEnv/rtSynth_rt/subjects/sub002/ses1/catPer")
# subject="catPer_000000sub002_1.txt"

# sub="sub002"
# ses="ses1"


# sub="sub001"
# ses="ses1"


# sub="sub002"
# ses="ses6"

# if ses=="ses6" or ses=="ses5":
#     catPerSession=2
# else:
#     catPerSession=1
# os.chdir(f"/Users/kailong/Desktop/rtEnv/rtSynth_rt/subjects/{sub}/{ses}/catPer")
# subject=f"catPer_000000{sub}_{catPerSession}.txt"

os.chdir(f"/Users/kailong/Desktop/rtEnv/rtSynth_rt/expScripts/catPer/data/")
# subject="catPer_123456subTest5.txt"
# subject="catPer_123456subTest6.txt"
# subject="catPer_123456sub0jeff.txt"
# subject="catPer_123456subTest7.txt"
sub="12345subShmily" #12345subTest11 12345subShmily
subject=f"catPer_{sub}.txt"


versionDict={'horizontal':['bedChair', 'tableBench'],
            'vertical':['benchBed', 'chairTable'],
            'diagonal':['chairBench', 'bedTable']}
version=checkVersion(subject)


f, ax = plt.subplots(1,2, figsize=(20, 10))
axis=versionDict[version][0]
(morph1acc, morph21acc, morph80acc, morph100acc,k,x0,exclusion,x,y)=logit(subject,axis,ax[0],sub)
axis=versionDict[version][1]
(morph1acc, morph21acc, morph80acc, morph100acc,k,x0,exclusion,x,y)=logit(subject,axis,ax[1],sub)

