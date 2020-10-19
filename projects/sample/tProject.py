"""-----------------------------------------------------------------------------

sample.py (Last Updated: 05/26/2020)

The purpose of this script is to actually to run the sample project.
Specifically, it will initiate a file watcher that searches for incoming dicom
files, do some sort of analysis based on the dicom file that's been received,
and then output the answer.

The purpose of this *particular* script is to demonstrated how you can use the
various scripts, functions, etc. we have developed for your use! The functions
we will reference live in 'rt-cloud/rtCommon/'.

Finally, this script is called from 'projectMain.py', which is called from
'run-projectInterface.sh'.

-----------------------------------------------------------------------------"""

# print a short introduction on the internet window
print(""
    "-----------------------------------------------------------------------------\n"
    "The purpose of this sample project is to demonstrate different ways you can\n"
    "implement functions, structures, etc. that we have developed for your use.\n"
    "You will find some comments printed on this html browser. However, if you want\n"
    "more information about how things work please talk a look at ‘sample.py’.\n"
    "Good luck!\n"
    "-----------------------------------------------------------------------------")

# import important modules
import os
import sys
import argparse
import numpy as np
import nibabel as nib
import scipy.io as sio
from subprocess import call
from nibabel.nicom import dicomreaders
import pydicom as dicom  # type: ignore
import time
import glob
import shutil
from nilearn.image import new_img_like
import joblib

print(''
    '|||||||||||||||||||||||||||| IGNORE THIS WARNING ||||||||||||||||||||||||||||')
from nibabel.nicom import dicomreaders
print(''
    '|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')

# obtain full path for current directory: '.../rt-cloud/projects/sample'
currPath = os.path.dirname(os.path.realpath(__file__))
# obtain full path for root directory: '.../rt-cloud'
rootPath = os.path.dirname(os.path.dirname(currPath))

# add the path for the root directory to your python path so that you can import
#   project modules from rt-cloud
sys.path.append(rootPath)
# import project modules from rt-cloud
from rtCommon.utils import loadConfigFile
from rtCommon.fileClient import FileInterface
import rtCommon.projectUtils as projUtils
from rtCommon.imageHandling import readRetryDicomFromFileInterface, getDicomFileName, convertDicomImgToNifti

# obtain the full path for the configuration toml file
defaultConfig = os.path.join(currPath, 'conf/sample.toml')

# realtime_ornt=nib.orientations.axcodes2ornt(('I', 'P', 'L'))
# ref_ornt=nib.orientations.axcodes2ornt(('P', 'S', 'L'))
# global ornt_transform
# ornt_transform = nib.orientations.ornt_transform(realtime_ornt,ref_ornt)


# tmp_folder='/tmp/kp578/'
# tmp_folder = '/gpfs/milgram/scratch60/turk-browne/kp578/sandbox/' # tmp_folder='/tmp/kp578/'
# if not os.path.isdir(tmp_folder): 
#     os.mkdir(tmp_folder)

def dicom2nii(dicomObject, filename,templateFunctionalVolume):
    niftiObject = dicomreaders.mosaic_to_nii(dicomObject)
    # print(nib.aff2axcodes(niftiObject.affine))
    temp_data = niftiObject.get_data()
    output_image_correct = nib.orientations.apply_orientation(temp_data, ornt_transform)
    correct_object = new_img_like(templateFunctionalVolume, output_image_correct, copy_header=True)
    print(nib.aff2axcodes(correct_object.affine))
    splitList=filename.split('/')
    # fullNiftiFilename="/".join(splitList[0:-1])+'/'+splitList[-1].split('.')[0]+'.nii.gz'
    fullNiftiFilename=os.path.join(tmp_folder, splitList[-1].split('.')[0]+'.nii.gz')
    print('fullNiftiFilename=',fullNiftiFilename)
    correct_object.to_filename(fullNiftiFilename)
    return fullNiftiFilename

# def printOrien(full_ref_BOLD):
#     ref_BOLD_obj = nib.load(full_ref_BOLD)
#     ref_bold_ornt = nib.aff2axcodes(ref_BOLD_obj.affine)
#     print('Ref BOLD orientation:')
#     print(ref_bold_ornt)
#     return ref_bold_ornt

def initializetProject(configFile,args):
    cfg=loadConfigFile(configFile)
    cfg.mode='tProject'
    cfg.dicomNamePattern="001_0000{}_000{}.dcm" #scanNum,TRnum    
    cfg.dicomDir = '/gpfs/milgram/project/realtime/DICOM/20201009.rtSynth_pilot001.rtSynth_pilot001/'
    cfg.runNum = [1] #[1, 2, 3, 4]
    cfg.scanNum = [5] #[5, 6 ,7 ,8]
    cfg.dataDir = '/gpfs/milgram/project/realtime/DICOM/20201009.rtSynth_pilot001.rtSynth_pilot001/'
    # cfg.station_stats = cfg.classifierDir + 'station_stats.npz'
    # cfg.subject_full_day_path = '{0}/{1}/{2}'.format(cfg.dataDir,cfg.bids_id,cfg.ses_id)
    # cfg.temp_nifti_dir = '{0}/converted_niftis/'.format(cfg.subject_full_day_path)
    # cfg.subject_reg_dir = '{0}/registration_outputs/'.format(cfg.subject_full_day_path)
    # cfg.nStations, cfg.stationsDict, cfg.last_tr_in_station, cfg.all_station_TRs = getStationInformation(cfg)

    # # REGISTRATION THINGS
    # cfg.wf_dir = '{0}/{1}/ses-{2:02d}/registration/'.format(cfg.dataDir,cfg.bids_id,1)
    # cfg.BOLD_to_T1= cfg.wf_dir + 'affine.txt'
    # cfg.T1_to_MNI= cfg.wf_dir + 'ants_t1_to_mniComposite.h5'
    # cfg.ref_BOLD=cfg.wf_dir + 'ref_image.nii.gz'

    # # GET CONVERSION FOR HOW TO FLIP MATRICES
    # cfg.axesTransform = getTransform()
    # ###### BUILD SUBJECT FOLDERS #######
    return cfg

def doRuns(cfg, fileInterface, projectComm):
    """
    This function is called by 'main()' below. Here, we use the 'fileInterface'
    to read in dicoms (presumably from the scanner, but here it's from a folder
    with previously collected dicom files), doing some sort of analysis in the
    cloud, and then sending the info to the web browser.

    INPUT:
        [1] cfg (configuration file with important variables)
        [2] fileInterface (this will allow a script from the cloud to access files
               from the stimulus computer, which receives dicom files directly
               from the Siemens console computer)
        [3] projectComm (communication pipe to talk with projectInterface)
    OUTPUT:
        None.

    This is the main function that is called when you run 'sample.py'.
    Here, you will set up an important argument parser (mostly provided by
    the toml configuration file), initiate the class fileInterface, and then
    call the function 'doRuns' to actually start doing the experiment.
    """

    # variables we'll use throughout
    # run = cfg.run[0] # what thr current run is 
    runNum = cfg.runNum[0]
    scanNum = cfg.scanNum[0]
    # before we get ahead of ourselves, we need to make sure that the necessary file
    #   types are allowed (meaning, we are able to read them in)... in this example,
    #   at the very least we need to have access to dicom and txt file types.
    # use the function 'allowedFileTypes' in 'fileClient.py' to check this!
    #   INPUT: None
    #   OUTPUT:
    #       [1] allowedFileTypes (list of allowed file types)

    allowedFileTypes = fileInterface.allowedFileTypes()
    print(""
    "-----------------------------------------------------------------------------\n"
    "Before continuing, we need to make sure that dicoms are allowed. To verify\n"
    "this, use the 'allowedFileTypes'.\n"
    "Allowed file types: %s" %allowedFileTypes) #['*']

    # obtain the path for the directory where the subject's dicoms live
    if cfg.isSynthetic:
        cfg.dicomDir = cfg.imgDir
    else:
        subj_imgDir = "{}.{}.{}".format(cfg.datestr, cfg.subjectName, cfg.subjectName)
        cfg.dicomDir = os.path.join(cfg.imgDir, subj_imgDir)
        cfg.dicomDir='/gpfs/milgram/project/realtime/DICOM/20201009.rtSynth_pilot001.rtSynth_pilot001/'
    print("Location of the subject's dicoms: \n%s\n" %cfg.dicomDir,
    "-----------------------------------------------------------------------------")

    # initialize a watch for the entire dicom folder (it doesn't look for a
    #   specific dicom) using the function 'initWatch' in 'fileClient.py'
    #   INPUT:
    #       [1] cfg.dicomDir (where the subject's dicom files live)
    #       [2] cfg.dicomNamePattern (the naming pattern of dicom files)
    #       [3] cfg.minExpectedDicomSize (a check on size to make sure we don't
    #               accidentally grab a dicom before it's fully acquired)
    print("• initalize a watch for the dicoms using 'initWatch'")
    fileInterface.initWatch(cfg.dicomDir, cfg.dicomNamePattern,
        cfg.minExpectedDicomSize)

    # we will use the function 'sendResultToWeb' in 'projectUtils.py' whenever we
    #   want to send values to the web browser so that they can be plotted in the
    #   --Data Plots-- tab
    #   INPUT:
    #       [1] projectComm (the communication pipe)
    #       [2] runNum (not to be confused with the scan number)
    #       [3] this_TR (timepoint of interest)
    #       [4] value (value you want to send over to the web browser)
    #       ** the inputs MUST be python integers; it won't work if it's a numpy int
    #
    # here, we are clearing an already existing plot
    print("• clear any pre-existing plot using 'sendResultToWeb'")
    projUtils.sendResultToWeb(projectComm, runNum, None, None)

    print(""
    "-----------------------------------------------------------------------------\n"
    "In this sample project, we will retrieve the dicom file for a given TR and\n"
    "then convert the dicom file to a nifti object. **IMPORTANT: In this sample\n"
    "we won't care about the exact location of voxel data (we're only going to\n"
    "indiscriminately get the average activation value for all voxels). This\n"
    "actually isn't something you want to actually do but we'll go through the\n"
    "to get the data in the appropriate nifti format in the 'advanced sample\n"
    "project.** We are doing things in this way because it is the simplest way\n"
    "we can highlight the functionality of rt-cloud, which is the purpose of\n"
    "this sample project.\n"
    ".............................................................................\n"
    "NOTE: We will use the function 'readRetryDicomFromFileInterface' to retrieve\n"
    "specific dicom files from the subject's dicom folder. This function calls\n"
    "'fileInterface.watchFile' to look for the next dicom from the scanner.\n"
    "Since we're using previously collected dicom data, this is functionality is\n"
    "not particularly relevant for this sample project but it is very important\n"
    "when running real-time experiments.\n"
    "-----------------------------------------------------------------------------\n")

    def getDicomFileName(cfg, scanNum, fileNum):
        """
        This function takes in different variables (which are both specific to the specific
        scan and the general setup for the entire experiment) to produce the full filename
        for the dicom file of interest.
        Used externally.
        """
        if scanNum < 0:
            raise ValidationError("ScanNumber not supplied or invalid {}".format(scanNum))

        # the naming pattern is provided in the toml file
        if cfg.dicomNamePattern is None:
            raise InvocationError("Missing config settings dicomNamePattern")

        if '{run' in cfg.dicomNamePattern:
            fileName = cfg.dicomNamePattern.format(scan=scanNum, run=fileNum)
        else:
            scanNumStr = str(scanNum).zfill(2)
            fileNumStr = str(fileNum).zfill(3)
            fileName = cfg.dicomNamePattern.format(scanNumStr, fileNumStr)
        fullFileName = os.path.join(cfg.dicomDir, fileName)

        return fullFileName


    Top_directory = '/gpfs/milgram/project/realtime/DICOM'
    # Top_directory = '/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/' # simulated folder for realtime folder where new incoming dicom files are pushed to

    ## - realtime feedback code
    # subject folder
    print('cfg=',cfg)
    # YYYYMMDD= '20201009' #'20201009' '20201015'
    # YYYYMMDD= '20201019' #'20201009' '20201015'
    # LASTNAME='rtSynth_pilot001'
    # PATIENTID='rtSynth_pilot001'
    YYYYMMDD=cfg.YYYYMMDD
    LASTNAME=cfg.realtimeFolder_subjectName
    PATIENTID=cfg.realtimeFolder_subjectName

    subjectFolder=f"{Top_directory}/{YYYYMMDD}.{LASTNAME}.{PATIENTID}/" #20190820.RTtest001.RTtest001: the folder for current subject # For each patient, a new folder will be generated:
    cfg.dicomDir=subjectFolder

    # tmp_folder='/tmp/kp578/'
    global tmp_folder
    tmp_folder=f'/gpfs/milgram/scratch60/turk-browne/kp578/{YYYYMMDD}.{LASTNAME}.{PATIENTID}/'
    # if os.path.isdir(tmp_folder):
    #   shutil.rmtree(tmp_folder)
    if not os.path.isdir(tmp_folder):
        os.mkdir(tmp_folder)

    import random
    randomlist = []
    for i in range(0,50):
        n = random.randint(1,19)
        randomlist.append(n)
    print(randomlist)

    # current TR dicom file name
    # SCANNUMBER='000001'
    # TRNUMBER='000001'
    # dicomFileName = f"001_{SCANNUMBER}_{TRNUMBER}.dcm" # DICOM_file #SCANNUMBER might be run number; TRNUMBER might be which TR is this currently.

    # this is the output of the recognition_dataAnalysis.py, meaning the day1 functional template volume in day1 anatomical space.
    # day1functionalInAnatomicalSpace='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/recognition/dataAnalysis/rtSynth_pilot001/nifti/day1functionalInAnatomicalSpace.nii.gz'

    realtime_ornt=nib.orientations.axcodes2ornt(('I', 'P', 'L'))
    ref_ornt=nib.orientations.axcodes2ornt(('P', 'S', 'L'))
    global ornt_transform
    ornt_transform = nib.orientations.ornt_transform(realtime_ornt,ref_ornt)

    # sub='pilot_sub001'
    sub=cfg.subjectName
    # ses=1
    ses=cfg.session
    # run='01'
    run=cfg.whichRun
    runNum=int(cfg.whichRun)

    homeDir="/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/" 
    dataDir=f"{homeDir}subjects/{sub}/ses{ses}_recognition/run{run}/nifti/"
    # templateFunctionalVolume=f'{dataDir}templateFunctionalVolume.nii.gz' #should be '/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/pilot_sub001/ses1_recognition/run01/nifti//templateFunctionalVolume.nii.gz'
    templateFunctionalVolume='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/pilot_sub001/ses1_recognition/run1/nifti//templateFunctionalVolume.nii.gz'
    # scanNum=1
    scanNum=int(cfg.whichScan)
    

    num_total_TRs=int(cfg.num_total_TRs)
    NFparams=[]
    for this_TR in np.arange(1,1+num_total_TRs):
        timeout_file = 5
        # time.sleep(2)
        fileName = getDicomFileName(cfg, scanNum, this_TR) # get the filename expected for the new DICOM file, might be f"{subjectFolder}{dicomFileName}"
        # fileName=subjectFolder+"001_000005_000149.dcm"
        print('fileName=',fileName)
        print("• use 'readRetryDicomFromFileInterface' to read dicom file for",
            "TR %d, %s" %(this_TR, fileName)) # fileName is a specific file name of the interested file
        dicomObject = readRetryDicomFromFileInterface(fileInterface, fileName,timeout_file) # wait till you find the next dicom is available
        # dicomObject=fileName
        newDicomFile=dicom2nii(dicomObject, fileName,templateFunctionalVolume) # convert dicom to nifti

        newDicomFile_aligned=tmp_folder+newDicomFile.split('/')[-1][0:-7]+'_aligned.nii.gz' #aligned to day1 functional template run volume in day1 anatomical space

        command = f"3dvolreg -base {templateFunctionalVolume} -prefix {newDicomFile_aligned} \
        -1Dfile {newDicomFile_aligned[0:-7]}_motion.1D -1Dmatrix_save {newDicomFile_aligned[0:-7]}_mat.1D {newDicomFile}"
        print('Running ' + command)
        A = time.time()
        call(command, shell=True)
        B = time.time()
        print('3dvolreg time=',B-A) 

        newDicomFile_aligned = nib.load(newDicomFile_aligned)
        newDicomFile_aligned = newDicomFile_aligned.get_data()
        newTR=newDicomFile_aligned.reshape(1,-1)
        print(newTR.shape)
        
        ## - load the saved model and apply it on the new coming dicom file.
        model_dir='/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/clf/'
        clf1 = joblib.load(model_dir+'pilot_sub001_bedchair_chairbench.joblib') 
        clf2 = joblib.load(model_dir+'pilot_sub001_bedchair_chairtable.joblib') 

        # then do this for each TR
        s1 = clf1.score(newTR, ['table'])
        s2 = clf2.score(newTR, ['table'])
        NFparam = np.mean([s1, s2]) # or an average or whatever
        print(NFparam)
        NFparams.append(NFparam)
        parameter = int(NFparam*10)+1 #random.randint(1,10)
        print('parameter=',parameter)
        
        ## - send the output of the model to web.
        projUtils.sendResultToWeb(projectComm, runNum, int(this_TR), parameter)
        # projUtils.sendResultToWeb(projectComm, NFparam, int(this_TR), parameter)


        # def convertToNifti(TRnum,scanNum,cfg,dicomData):
        #     #anonymizedDicom = anonymizeDicom(dicomData) # should be anonymized already
        #     scanNumStr = str(scanNum).zfill(2)
        #     fileNumStr = str(TRnum).zfill(3)
        #     expected_dicom_name = cfg.dicomNamePattern.format(scanNumStr,fileNumStr)
        #     tempNiftiDir = os.path.join(cfg.dataDir, 'tmp/convertedNiftis/')
        #     nameToSaveNifti = expected_dicom_name.split('.')[0] + '.nii.gz'
        #     fullNiftiFilename = os.path.join(tempNiftiDir, nameToSaveNifti)
        #     print('fullNiftiFilename=',fullNiftiFilename)
        #     if not os.path.isfile(fullNiftiFilename): # only convert if haven't done so yet (check if doesn't exist)
        #        fullNiftiFilename = dnh.saveAsNiftiImage(dicomData,expected_dicom_name,cfg)
        #     else:
        #         print('SKIPPING CONVERSION FOR EXISTING NIFTI {}'.format(fullNiftiFilename))
        #     return fullNiftiFilename
        #     # ask about nifti conversion or not

        # if cfg.isSynthetic:
        #     niftiObject = convertDicomImgToNifti(dicomData)
        # else:
        #     # use 'dicomreaders.mosaic_to_nii' to convert the dicom data into a nifti
        #     #   object. additional steps need to be taken to get the nifti object in
        #     #   the correct orientation, but we will ignore those steps here. refer to
        #     #   the 'advanced sample project' for more info about that
        #     print("| convert dicom data into a nifti object")
        #     niftiObject = dicomreaders.mosaic_to_nii(dicomData)

        #     fullNiftiFilename = convertToNifti(this_TR,scanNum,cfg,dicomData)

        # all_avg_activations[this_TR]=parameter
        # # create the full path filename of where we want to save the activation values vector
        # #   we're going to save things as .txt and .mat files
        # output_textFilename = tmp_folder+'avg_activations.txt'
        # output_matFilename = tmp_folder+'avg_activations.mat'
        # fileInterface.putTextFile(output_textFilename,str(all_avg_activations))
        # sio.savemat(output_matFilename,{'value':all_avg_activations})
    # use 'putTextFile' from 'fileClient.py' to save the .txt file
    #   INPUT:
    #       [1] filename (full path!)
    #       [2] data (that you want to write into the file)
    print(""
    "-----------------------------------------------------------------------------\n"
    "• save activation value as a text file to tmp folder")
        

    # use sio.save mat from scipy to save the matlab file
    print("• save activation value as a matlab file to tmp folder")
    

    print(""
    "-----------------------------------------------------------------------------\n"
    "REAL-TIME EXPERIMENT COMPLETE!")

    return


def main(argv=None):
    """
    This is the main function that is called when you run 'sample.py'.

    Here, you will load the configuration settings specified in the toml configuration
    file, initiate the class fileInterface, and then call the function 'doRuns' to
    actually start doing the experiment.
    """

    # define the parameters that will be recognized later on to set up fileIterface
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                           help='experiment config file (.json or .toml)')
    argParser.add_argument('--runs', '-r', default='', type=str,
                           help='Comma separated list of run numbers')
    argParser.add_argument('--scans', '-s', default='', type=str,
                           help='Comma separated list of scan number')
    # This parameter is used for projectInterface
    argParser.add_argument('--commpipe', '-q', default=None, type=str,
                           help='Named pipe to communicate with projectInterface')
    argParser.add_argument('--filesremote', '-x', default=False, action='store_true',
                           help='retrieve dicom files from the remote server')
    args = argParser.parse_args(argv)

    # load the experiment configuration file
    # cfg = loadConfigFile(args.config)
    cfg = initializetProject(args.config,args)
    # obtain paths for important directories (e.g. location of dicom files)
    if cfg.imgDir is None:
        cfg.imgDir = os.path.join(currPath, 'dicomDir')
    cfg.codeDir = currPath

    # open up the communication pipe using 'projectInterface'
    projectComm = projUtils.initProjectComm(args.commpipe, args.filesremote)

    # initiate the 'fileInterface' class, which will allow you to read and write
    #   files and many other things using functions found in 'fileClient.py'
    #   INPUT:
    #       [1] args.filesremote (to retrieve dicom files from the remote server)
    #       [2] projectComm (communication pipe that is set up above)
    fileInterface = FileInterface(filesremote=args.filesremote, commPipes=projectComm)

    # now that we have the necessary variables, call the function 'doRuns' in order
    #   to actually start reading dicoms and doing your analyses of interest!
    #   INPUT:
    #       [1] cfg (configuration file with important variables)
    #       [2] fileInterface (this will allow a script from the cloud to access files
    #               from the stimulus computer that receives dicoms from the Siemens
    #               console computer)
    #       [3] projectComm (communication pipe to talk with projectInterface)
    doRuns(cfg, fileInterface, projectComm)

    return 0


if __name__ == "__main__":
    """
    If 'sample.py' is invoked as a program, then actually go through all of the portions
    of this script. This statement is not satisfied if functions are called from another
    script using "from sample.py import FUNCTION"
    """
    main()
    sys.exit(0)
