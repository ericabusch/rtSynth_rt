import rtCommon.utils as utils
import os
# sys.path.append('/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/')
def mkdir(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

def cfg_loading(toml=''):
    def findDir(path):
        from glob import glob
        # _path = glob(path)[0]+'/'
        _path = glob(path)
        if len(_path)==0: # if the dir is not found. get rid of the "*" and return
            _path=path.split("*")
            _path=''.join(_path)
        else:
            _path = _path[0]+'/'
        return _path

    # toml="pilot_sub001.ses1.toml"
    # cfg = utils.loadConfigFile(f"/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/rtSynth_rt/conf/{toml}")
        
    if 'watts' in os.getcwd():
        cfg = utils.loadConfigFile(f"/home/watts/Desktop/ntblab/kailong/rtSynth_rt/projects/rtSynth_rt/conf/{toml}")
        cfg.projectDir="/home/watts/Desktop/ntblab/kailong/rtSynth_rt/"
    elif 'kailong' in os.getcwd():
        cfg = utils.loadConfigFile(f"/Users/kailong/Desktop/rtEnv/rtSynth_rt/projects/rtSynth_rt/conf/{toml}")
        cfg.projectDir="/Users/kailong/Desktop/rtEnv/rtSynth_rt/"
    elif 'milgram' in os.getcwd():
        cfg = utils.loadConfigFile(f"/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/projects/rtSynth_rt/conf/{toml}")
        cfg.projectDir="/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/"
    else: 
        raise Exception('path error')
    
    cfg.tmp_folder="/gpfs/milgram/scratch60/turk-browne/kp578/rtcloud_rt/"
    cfg.orderFolder=f'{cfg.projectDir}expScripts/recognition/orders/'
    cfg.subjects_dir=f'{cfg.projectDir}subjects/'
    cfg.dicom_folder="/gpfs/milgram/project/realtime/DICOM/"
    cfg.recognition_expScripts_dir = f"{cfg.projectDir}expScripts/recognition/"

    cfg.TR=2

    cfg.preDay_dicom_dir  = findDir(f"{cfg.dicom_folder}{cfg.preDay_YYYYMMDD}.{cfg.LASTNAME}*.{cfg.LASTNAME}*/")  #e.g. /gpfs/milgram/project/realtime/DICOM/20201009.rtSynth_pilot001.rtSynth_pilot001/  # cfg.preDay_YYYYMMDD is "0" when there is no previous day
    cfg.dicom_dir         = findDir(f"{cfg.dicom_folder}{cfg.YYYYMMDD}.{cfg.LASTNAME}*.{cfg.LASTNAME}*/")  # YYYYMMDD.$LASTNAME.$PATIENTID  e.g. /gpfs/milgram/project/realtime/DICOM/20201019.rtSynth_pilot001_2.rtSynth_pilot001_2/ inside which is like 001_000003_000067.dcm
    cfg.recognition_dir   = f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/recognition/"
    cfg.feedback_dir      = f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/feedback/"
    cfg.usingModel_dir    = f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session-1}/recognition/clf/"
    cfg.trainingModel_dir = f"{cfg.subjects_dir}{cfg.subjectName}/ses{cfg.session}/recognition/clf/"
    cfg.templateFunctionalVolume = f"{cfg.recognition_dir}/templateFunctionalVolume.nii" 
    cfg.dicomNamePattern  = "001_0000{}_000{}.dcm"
    cfg.mask_dir          = f"{cfg.recognition_dir}mask/"
    cfg.chosenMask = f"{cfg.subjects_dir}{cfg.subjectName}/ses1/recognition/chosenMask.nii.gz"

    # prepare folder structure
    mkdir(f"{cfg.subjects_dir}{cfg.subjectName}") # mkdir subject folder
    for curr_ses in [1,5]:
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/catPer/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/anat/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/recognition/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/recognition/clf/")
    
    for curr_ses in [2,3,4]:
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/recognition/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/recognition/clf/")
        mkdir(f"{cfg.subjects_dir}{cfg.subjectName}/ses{curr_ses}/feedback/")

    return cfg



# if os.path.isdir(tmp_folder):
#   shutil.rmtree(tmp_folder)
