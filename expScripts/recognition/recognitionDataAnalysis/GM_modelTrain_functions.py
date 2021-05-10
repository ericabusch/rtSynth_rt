import os,time,shutil
import numpy as np
from glob import glob
def find_ABCD_T1w_MPR_vNav(sub):
        #这个函数的功能是找到第二个ABCD_T1w_MPR_vNav   usable的前面的数字，保存在ABCD_T1w_MPR_vNav.txt里面
        
        raw_dir="/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/"
        os.chdir(raw_dir)
        f = open(f"{raw_dir}{sub}_run_name.txt", "r") # print(f.read())
        d = f.read()
        t = d.split("ABCD_T1w_MPR_vNav")[3]
        t = int(t.split("\n")[1])
        f = open(f"{raw_dir}{sub}_ABCD_T1w_MPR_vNav.txt","w")
        f.write(f"T1_ID={t}")
        f.close()

def wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))
        
def find_T1_in_niiFolder(T1_ID,sub):
    subjectName=sub.split("_")[1] # sub 可能是rtSynth_sub001_ses5 或者 rtSynth_sub001; subjectName 应该是sub001之类的
    if len(sub.split("_"))==3:
        ses=int(sub.split("_")[2].split("ses")[1]) #ses应该是数字
    else:
        ses=1

    nii_path=f"/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/{sub}/nifti/"
    print(f"nii_path={nii_path}")
    files = glob(f"{nii_path}*.nii")
    for curr_file in files:
        runID=curr_file.split("/")[-1].split("_")[-1].split(".nii")[0]
        try:
            if int(runID)==8:
                print(curr_file)
                T1=curr_file
        except:
            pass
    AnatFolder = f"/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/{subjectName}/ses{ses}/anat/"
    print(f"T1={T1}")
    print(f"AnatFolder={AnatFolder}")
    shutil.copyfile(T1,f"{AnatFolder}T1.nii")

def _split(sub):
    subjectName=sub.split("_")[1] # sub 可能是rtSynth_sub001_ses5 或者 rtSynth_sub001; subjectName 应该是sub001之类的
    if len(sub.split("_"))==3:
        ses=int(sub.split("_")[2].split("ses")[1]) #ses应该是数字
    else:
        ses=1

    code_dir="/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/"
    f = open(f"{code_dir}{sub}_subjectName.txt","w")
    f.write(f"subjectName={subjectName} ; ses={ses}")
    f.close()
