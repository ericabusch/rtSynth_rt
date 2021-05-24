# import urllib.request
# import numpy as np
# from tqdm import tqdm
# # select output dir
# outdir = './carchair'
# stride = 1

# cats = ['furniture', 'vehicles']
# furn = ['bedChair', 'bedTable', 'benchBed', 'chairBench', 'chairTable', 'tableBench']
# veh = ['limoToSUV', 'limoToSedan', 'limoToSmart', 'smartToSedan', 'suvToSedan', 'suvToSmart']
# viewpoints = np.arange(0, 40, stride)
# morphs = [np.array([1,100]), np.array([10,99])] #[np.arange(1,101, stride), np.arange(10,100,stride)]
# blendDict = dict(zip(cats, morphs))
# axisDict = dict(zip(cats, [furn, veh]))
# urlstem = "https://s3.amazonaws.com/morphrecog-images-1/{}_{}_{}.png.png"

# for cat in cats:
#     blends = blendDict[cat]
#     axes = axisDict[cat]
#     for axis in axes[:]:
#         for blend in blends[:]:
#             for view in viewpoints[:]:
#                 print(urlstem.format(axis, blend, view))
#                 urllib.request.urlretrieve(urlstem.format(axis, blend, view),
#                                            "{}/{}_{}_{}.png".format(outdir, axis, blend, view))




import urllib.request
import numpy as np
from tqdm import tqdm
import os
outdir = './carchair/'
os.chdir("/Users/kailong/Desktop/rtEnv/rtSynth_rt/expScripts/catPer/")
cats = ['furniture']
furn = [['bedChair', 'tableBench']]
stride=3
# veh = ['limoToSUV', 'limoToSedan', 'limoToSmart', 'smartToSedan', 'suvToSedan', 'suvToSmart']
viewpoints = np.arange(0, 40, stride)
morphs = [np.array([18, 26, 34, 42, 50, 58, 66, 74, 82])] #[np.arange(1,101, stride), np.arange(10,100,stride)]
blendDict = dict(zip(cats, morphs))
axisDict = dict(zip(cats, furn))
urlstem = "https://s3.amazonaws.com/morphrecog-images-1/{}_{}_{}.png.png" #https://s3.amazonaws.com/morphrecog-images-1/bedChair_18_0.png.png


for cat in cats:
    blends = blendDict[cat]
    axes = axisDict[cat]
    for axis in axes[:]:
        for blend in blends[:]:
            for view in viewpoints[:]:
                print(urlstem.format(axis, blend, view))
                urllib.request.urlretrieve(urlstem.format(axis, blend, view),
                                           "{}/{}_{}_{}.png".format(outdir, axis, blend, view))
