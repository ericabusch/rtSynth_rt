
# modules and functions
import pandas as pd
import numpy as np
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
import itertools
from scipy import stats
from glob import glob
tag='condition5'
def loadNpInDf(fileName):
#     main_dir='/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/FilterTesting/testMiniclass/'
    main_dir = "/gpfs/milgram/project/turk-browne/projects/rtSynth/alex_scratch/realtime/"
    return np.load(main_dir+fileName+'.npy')

# def preloadDfnumpy(testEvidence,List=['AC_A_evidence','AD_A_evidence','AC_B_evidence','AD_B_evidence','A_evidence_forATrials','A_evidence_forBTrials']):
def preloadDfnumpy(testEvidence,List=['A_evidence_forATrials','A_evidence_forBTrials']):
    # this function convert the dataframe cell numpy array into real numpy array, was a string pointing to a file
    import warnings
    warnings.filterwarnings("ignore")
    for i in range(len(testEvidence)):
        for L in List:
            testEvidence[L].iloc[i]=loadNpInDf(testEvidence[L].iloc[i])
    warnings.filterwarnings("default")
    return testEvidence

def _and_(L):
    if len(L)==2:
        return np.logical_and(L[0],L[1])
    else:
        return np.logical_and(L[0],_and_(L[1:]))

def resample(L,iters=1000):
    L=np.asarray(L).reshape(-1)
    sample_mean=[]
    for iter in range(iters):
        resampleID=np.random.choice(L.shape[0], L.shape[0], replace=True)
        resample_acc=L[resampleID]
        sample_mean.append(np.nanmean(resample_acc))
    sample_mean=np.asarray(sample_mean)
    m = np.nanmean(sample_mean,axis=0)
    upper=np.percentile(sample_mean, 97.5, axis=0)
    lower=np.percentile(sample_mean, 2.5, axis=0)
    return m,m-lower,upper-m

def barplot_annotate_brackets(num1, num2, data, center, height,yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        text = ''
        p = .05

        while data < p:
            if len(text)>=3:
                break
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)
    return 

def bar(LL,labels=None,title=None,pairs=None,pvalue=None):
    import matplotlib.pyplot as plt
    D=np.asarray([resample(L) for L in LL])
    m=D[:,0]
    lower=D[:,1]
    upper=D[:,2]
    x_pos = np.arange(len(labels))
    if len(LL)>20:
        figsize=30
    else:
        figsize=10
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    ax.bar(x_pos, m, yerr=[lower,upper], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('object evidence')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.yaxis.grid(True)
    # plt.tight_layout()
    plt.xticks(rotation=30,ha='right')
    if pairs!=None:
        for pair in pairs:
            barplot_annotate_brackets(pair[0], pair[1], pvalue[pair], x_pos, m)
            m[pair[0]]+=0.1
            m[pair[1]]+=0.1
    plt.show()
    return m,lower,upper,ax

def assertKeys(t0,t1,keys=['testRun','targetAxis','obj','otherObj']):
    # this function compare the given keys of the given two df and return true if they are exactly the same
    for key in keys:
        if not np.all(np.asarray(t1[key])==np.asarray(t0[key])):
            return False
    return True

def concatArrayArray(c): #[array[],array[]]
    ct=[]
    List=[list(j) for j in c] # transform [array[],array[]] to [list[],list[]]
    for i in range(len(c)):
        ct=ct+List[i] # concatenate List
    return ct

def _ttest_(x,y):
    x=np.asarray(x)
    y=np.asarray(y)
    x_=~np.isnan(x)
    y_=~np.isnan(y)
    x=x[np.logical_and(x_,y_)]
    y=y[np.logical_and(x_,y_)]
    mmmm=stats.ttest_rel(x,y)[1]
    return mmmm

# load saved results
accuracyContainer=[]
testEvidence=[]
model_parent_dir = '/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/alex_clf3'
filterTypes = ['noFilter','highPassRealTime','highPassBetweenRuns',
                    'filter_all_TRs_no_EM', 'filter_all_TRs_no_EM_50','filter_all_TRs_no_EM_70',
                    'filter_all_TRs_no_EM_90', 'filter_all_TRs_no_EM_110','filter_all_TRs_no_EM_130',
                    'filter_all_TRs_no_EM_150','filter_all_TRs_no_EM_170', 'filter_all_TRs_no_EM_190',
                   'filter_all_TRs_no_EM_210', 'filter_all_TRs_no_EM_230', 'filter_all_TRs_no_EM_250',
                   'filter_all_TRs_no_EM_270'
                    ]
val = [job.split("/")[-1] for job in glob("/gpfs/milgram/scratch60/turk-browne/an633/filter_all_TRs_scale_EM_*")]
filterTypes += ["filter_all_TRs_scale_EM_" + scale for scale in sorted([(v.split("_")[-1]) for v in val])]



for include in tqdm([0.1,0.3,0.6,0.9,1]):
    for roi in ['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']:
        for filterType in filterTypes :
            for testRun in [1,2,3,4,5,6]:
                # if filterType=='KalmanFilter_filter_analyze_voxel_by_voxel':
                model_folder = f'{model_parent_dir}/{np.float(include)}/{roi}/{filterType}/{testRun}/{tag}/'
                # else:
                #     model_folder = f'/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/clf/{np.float(include)}/{roi}/{filterType}/{testRun}/'
                try:
                    accuracyContainer.append(pd.read_csv(f"{model_folder}accuracy.csv"))
                    testEvidence.append(pd.read_csv(f'{model_folder}testEvidence.csv'))
                except:
                    pass
accuracyContainer=pd.concat(accuracyContainer, ignore_index=True)
testEvidence=pd.concat(testEvidence, ignore_index=True)


# filterTypes=['noFilter', 'highPassRealTime', 'highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']

subjects=np.unique(accuracyContainer['sub'])
ROIs=['V1', 'fusiform', 'IT', 'LOC', 'occitemp', 'parahippo']


# def evidenceAcrossFiltertypes(ROI="V1",paired_ttest=False):
#     # construct a list where the first one is 'A_evidence_forATrials for noFilter', second is 'A_evidence_forBTrials for noFilter', third is empty, 4th is 'A_evidence_forATrials for highpass' and so on
#     # for each element of the list, take 'A_evidence_forATrials for noFilter' for example. This is 1440*32=46080 numbers (say we have 32 subjects), each number is raw value of the 'A_evidence_forATrials for noFilter' for that subject.
#     a=[]
#     labels=[]
#     for i in range(len(filterTypes)): # for each filterType, each subject has one value for A_evidence_forATrials and another value for A_evidence_forBTrials
#         c=[]
#         d=[]

#         # to get one single number for A_evidence_forATrials for each subject. 
#         # you will need to extract the corresponding conditions and conbine the data together. 
#         for sub in subjects:
#             t=testEvidence[_and_([ #extract
#                 testEvidence['roi']==ROI,
#                 testEvidence['filterType']==filterTypes[i],
#                 testEvidence['include']==1.,
#                 testEvidence['sub']==sub
#             ])]
#             t=preloadDfnumpy(t)

#             c.append(np.asarray(list(t['A_evidence_forATrials'])).reshape(-1)) #conbine the data together
#             d.append(np.asarray(list(t['A_evidence_forBTrials'])).reshape(-1))

#         a.append(concatArrayArray(c))
#         a.append(concatArrayArray(d))
#         a.append([])
#         labels.append(filterTypes[i] + ' A_evidence_forATrials')
#         labels.append(filterTypes[i] + ' A_evidence_forBTrials')
#         labels.append('')
#     print('len of a=',[len(i) for i in a])
#     # paired t-test
#     objects=np.arange(5)
#     allpairs = itertools.combinations(objects,2)
#     pvalue={}
#     pairs=[]
#     for pair in allpairs:
#         i=pair[0]
#         j=pair[1]
#         if paired_ttest==True:
#             print(f"{filterTypes[i]} {filterTypes[j]} p={_ttest_(a[i*3],a[j*3])}")
#             pvalue[(i*3,j*3)]=_ttest_(a[i*3],a[j*3])
#         else:
#             print(f"{filterTypes[i]} {filterTypes[j]} p={_ttest_(a[i*3],a[j*3])}")
#             pvalue[(i*3,j*3)]=_ttest_(a[i*3],a[j*3])

#         pairs.append((i*3,j*3))

#     bar(a,labels=labels,title=f'raw evidence for each trial: across filterTypes, objEvidence and other Evidence, within only {ROI}, include=1. paired_ttest={paired_ttest}',pairs=pairs,pvalue=pvalue)

#     e=[np.asarray(a[i])[~np.isnan(np.asarray(a[i]))] for i in range(len(a))]
#     _=plt.boxplot(e)

# for i in range(len(ROIs)):
#     evidenceAcrossFiltertypes(ROI=ROIs[i])


def evidenceAcrossFiltertypes_meanForSub(ROI="V1",paired_ttest=True):
    # construct a list where the first one is 'A_evidence_forATrials for noFilter', second is 'A_evidence_forBTrials for noFilter', third is empty, 4th is 'A_evidence_forATrials for highpass' and so on
    # for each element of the list, take 'A_evidence_forATrials for noFilter' for example. This is 32 numbers (say we have 32 subjects), each number is mean value of the 'A_evidence_forATrials for noFilter' for that subject.

    # across filterType, take the difference between objEvidence and other Evidence, within only V1, include=1.
    # filterTypes=['noFilter', 'highPassRealTime', 'highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']

    # I want to construct a list where the first one is 'A_evidence_forATrials for noFilter', second is 'A_evidence_forBTrials for noFilter', third is empty, 4th is 'A_evidence_forATrials for highpass' and so on
    # for each element of the list, take 'A_evidence_forATrials for noFilter' for example. This is 32 numbers (say we have 32 subjects), each number is the mean value of the 'A_evidence_forATrials for noFilter' for that subject.
    a=[]
    labels=[]
    for i in range(len(filterTypes)): # for each filterType, each subject has one value for A_evidence_forATrials and another value for A_evidence_forBTrials
        c=[]
        d=[]

        # to get one single number for A_evidence_forATrials for each subject. 
        # you will need to extract the corresponding conditions and conbine the data together. 
        for sub in subjects:
            t=testEvidence[_and_([ #extract
                testEvidence['roi']==ROI,
                testEvidence['filterType']==filterTypes[i],
                testEvidence['include']==1.,
                testEvidence['sub']==sub
            ])]
            t=preloadDfnumpy(t)

            c.append(np.nanmean(np.asarray(list(t['A_evidence_forATrials'])))) #conbine the data together
            d.append(np.nanmean(np.asarray(list(t['A_evidence_forBTrials']))))

        a.append(c)
        a.append(d)
        a.append([])
        labels.append(filterTypes[i] + ' A_evidence_forATrials')
        labels.append(filterTypes[i] + ' A_evidence_forBTrials')
        labels.append('')
    print('len of a = ',[len(i) for i in a])

    e=[np.asarray(a[i])[~np.isnan(np.asarray(a[i]))] for i in range(len(a))]
    _=plt.boxplot(e)

    # paired t-test
    objects=np.arange(5)
    allpairs = itertools.combinations(objects,2)
    pvalue={}
    pairs=[]
    for pair in allpairs:
        i=pair[0]
        j=pair[1]
        if paired_ttest==True:
            print(f"{filterTypes[i]} {filterTypes[j]} p={_ttest_(a[i*3],a[j*3])}")
            pvalue[(i*3,j*3)]=_ttest_(a[i*3],a[j*3])
        else:
            print(f"{filterTypes[i]} {filterTypes[j]} p={_ttest_(a[i*3],a[j*3])}")
            pvalue[(i*3,j*3)]=_ttest_(a[i*3],a[j*3])

        pairs.append((i*3,j*3))

    bar(a,labels=labels,title=f'mean evidence for each subject: across filterTypes, objEvidence and other Evidence, within only {ROI}, include=1. paired_ttest={paired_ttest}',pairs=pairs,pvalue=pvalue)

for i in range(0, len(ROIs)):
    evidenceAcrossFiltertypes_meanForSub(ROI=ROIs[i])


def accuracyAcrossFiltertype(ROI="V1",paired_ttest=False):
    # accuracy: across filterType, take subject mean, within only V1, include=1.

    # I want to construction a list whose 1st element is the accuracy for noFilter, 2nd for highpass and so on.
    # each element is 32 numbers for 32 subjects. each number is the mean accuracy for that subject.
    a=[]
    for i in range(len(filterTypes)):
        b=[]
        for sub in tqdm(subjects):
            try:
                b.append(np.mean(accuracyContainer[
                        _and_([
                            accuracyContainer['roi']==ROI, 
                            accuracyContainer['filterType']==filterTypes[i],
                            accuracyContainer['sub']==int(sub),
                            accuracyContainer['include']==1.
                        ])]['acc']))
            except:
                pass
        a.append(np.asarray(b))
    # bar(a,labels=list(filterTypes),title=f'accuracy: across filterTypes, within only {ROI}, include=1.')
    e=[np.asarray(a[i])[~np.isnan(np.asarray(a[i]))] for i in range(len(a))]
    _=plt.boxplot(e)

    # paired t-test
    objects=np.arange(5)
    allpairs = itertools.combinations(objects,2)
    pvalue={}
    pairs=[]
    for pair in allpairs:
        i=pair[0]
        j=pair[1]
        if paired_ttest==True:
            print(f"{filterTypes[i]} {filterTypes[j]} p={_ttest_(a[i],a[j])}")
            pvalue[(i,j)]=_ttest_(a[i],a[j])
        else:
            print(f"{filterTypes[i]} {filterTypes[j]} p={_ttest_(a[i],a[j])}")
            pvalue[(i,j)]=_ttest_(a[i],a[j])

        pairs.append((i,j))
    bar(a,labels=list(filterTypes),title=f'accuracy: across filterTypes, within only {ROI}, include=1.  paired_ttest={paired_ttest}',pairs=pairs,pvalue=pvalue)
    return a

for i in range(0, len(ROIs)):
    accuracyAcrossFiltertype(ROI=ROIs[i])

filterTypes=['noFilter',
        'highPassRealTime',
        'highPassBetweenRuns',
        'filter_all_TRs_no_EM_270',
        'filter_all_TRs_scale_EM_0.5563312669021389'
        ]










def evidenceAcrossregularizationTags_meanForSub(ROI="V1",paired_ttest=True):
    # construct a list where the first one is 'A_evidence_forATrials for noFilter', second is 'A_evidence_forBTrials for noFilter', third is empty, 4th is 'A_evidence_forATrials for highpass' and so on
    # for each element of the list, take 'A_evidence_forATrials for noFilter' for example. This is 32 numbers (say we have 32 subjects), each number is mean value of the 'A_evidence_forATrials for noFilter' for that subject.

    # across filterType, take the difference between objEvidence and other Evidence, within only V1, include=1.
    # filterTypes=['noFilter', 'highPassRealTime', 'highPassBetweenRuns','KalmanFilter_filter_analyze_voxel_by_voxel']

    # I want to construct a list where the first one is 'A_evidence_forATrials for noFilter', second is 'A_evidence_forBTrials for noFilter', third is empty, 4th is 'A_evidence_forATrials for highpass' and so on
    # for each element of the list, take 'A_evidence_forATrials for noFilter' for example. This is 32 numbers (say we have 32 subjects), each number is the mean value of the 'A_evidence_forATrials for noFilter' for that subject.
    a=[]
    labels=[]
    # for i in range(len(filterTypes)): # for each filterType, each subject has one value for A_evidence_forATrials and another value for A_evidence_forBTrials
    for regularization_tag_id in range(len(regularization_tags)):
        regularization_tag=regularization_tags[regularization_tag_id]

        c=[]
        d=[]

        # to get one single number for A_evidence_forATrials for each subject. 
        # you will need to extract the corresponding conditions and conbine the data together. 
        for sub in subjects:
            t=testEvidence[_and_([ #extract
                testEvidence['roi']==ROI,
                testEvidence['filterType']==filterTypes[0],
                testEvidence['regularization_tag']==regularization_tag,
                testEvidence['sub']==sub
            ])]
            t=preloadDfnumpy(t)

            c.append(np.nanmean(np.asarray(list(t['A_evidence_forATrials'])))) #conbine the data together
            d.append(np.nanmean(np.asarray(list(t['A_evidence_forBTrials']))))

        a.append(c)
        a.append(d)
        a.append([])
        labels.append(regularization_tag + ' A_evidence_forATrials')
        labels.append(regularization_tag + ' A_evidence_forBTrials')
        labels.append('')
    print('len of a = ',[len(i) for i in a])

    e=[np.asarray(a[i])[~np.isnan(np.asarray(a[i]))] for i in range(len(a))]
    _=plt.boxplot(e)

    # paired t-test
    objects=np.arange(5)
    allpairs = itertools.combinations(objects,2)
    pvalue={}
    pairs=[]
    for pair in allpairs:
        i=pair[0]
        j=pair[1]
        if paired_ttest==True:
            print(f"{filterTypes[i]} {filterTypes[j]} p={_ttest_(a[i*3],a[j*3])}")
            pvalue[(i*3,j*3)]=_ttest_(a[i*3],a[j*3])
        else:
            print(f"{filterTypes[i]} {filterTypes[j]} p={_ttest_(a[i*3],a[j*3])}")
            pvalue[(i*3,j*3)]=_ttest_(a[i*3],a[j*3])

        pairs.append((i*3,j*3))

    bar(a,labels=regularization_tags,title=f'mean evidence for each subject: across filterTypes, objEvidence and other Evidence, within only {ROI}, include=1. paired_ttest={paired_ttest}',pairs=pairs,pvalue=pvalue)

for i in range(0, len(ROIs)):
    evidenceAcrossFiltertypes_meanForSub(ROI=ROIs[i])
