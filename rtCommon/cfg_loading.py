import rtCommon.utils as utils
import os
# sys.path.append('/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/')
def mkdir(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

def cfg_loading(toml=''):
    # toml="pilot_sub001.ses1.toml"
    # cfg = utils.loadConfigFile(f"/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/rtSynth_rt/conf/{toml}")
    cfg = utils.loadConfigFile(f"/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/projects/rtSynth_rt/conf/{toml}")
    

    cfg.dicom_dir         = f"{cfg.dicom_folder}{cfg.YYYYMMDD}.{cfg.LASTNAME}.{cfg.LASTNAME}/"
    cfg.recognition_dir   = f"{cfg.subjects_dir}/{cfg.subjectName}/ses{cfg.session}/recognition/"
    cfg.feedback_dir      = f"{cfg.subjects_dir}/{cfg.subjectName}/ses{cfg.session}/feedback/"
    cfg.usingModel_dir    = f"{cfg.subjects_dir}/{cfg.subjectName}/ses{cfg.session-1}/recognition/clf/"
    cfg.trainingModel_dir = f"{cfg.subjects_dir}/{cfg.subjectName}/ses{cfg.session}/recognition/clf/"

    # prepare folder structure
    for curr_ses in [1,5]:
        mkdir(f"subjects/{cfg.subjectName}/ses{curr_ses}/")
        mkdir(f"subjects/{cfg.subjectName}/ses{curr_ses}/catPer/")
        mkdir(f"subjects/{cfg.subjectName}/ses{curr_ses}/anat/")
        mkdir(f"subjects/{cfg.subjectName}/ses{curr_ses}/recognition/")
        mkdir(f"subjects/{cfg.subjectName}/ses{curr_ses}/recognition/clf/")
    

    for curr_ses in [2,3,4]:
        mkdir(f"subjects/{cfg.subjectName}/ses{curr_ses}/")
        mkdir(f"subjects/{cfg.subjectName}/ses{curr_ses}/recognition/")
        mkdir(f"subjects/{cfg.subjectName}/ses{curr_ses}/recognition/clf/")
        mkdir(f"subjects/{cfg.subjectName}/ses{curr_ses}/feedback/")


    return cfg



# if os.path.isdir(tmp_folder):
#   shutil.rmtree(tmp_folder)
