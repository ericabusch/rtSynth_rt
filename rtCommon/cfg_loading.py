
import rtCommon.utils as utils
import os

def mkdir(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

def cfg_loading(toml=''):
    # toml="pilot_sub001.ses1.toml"
    cfg = utils.loadConfigFile(f"/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/rtSynth_rt/conf/{toml}")

    cfg.dicom_dir=f"{cfg.dicom_folder}{cfg.YYYYMMDD}.{cfg.LASTNAME}/"
    cfg.recognition_dataFolder=f"{cfg.subjects_dir}/{cfg.subjectName}/{cfg.session}_recognition/"
    cfg.feedback_dataFolder=f"{cfg.subjects_dir}/{cfg.subjectName}/{cfg.session}_feedback/"
    cfg.model_dir=f"{cfg.subjects_dir}/{cfg.subjectName}/{cfg.session}_recognition/clf/"
    
    return cfg