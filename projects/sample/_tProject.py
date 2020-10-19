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
from rtCommon.structDict import StructDict

# obtain the full path for the configuration toml file
defaultConfig = os.path.join(currPath, 'conf/tProject.toml')

def dicom2nii(filename):
    # filename=dicomFolder+'001_000001_000018.dcm'
    # dicomObject = dicom.read_file(dicomFolder+filename+'.dcm')
    dicomObject = dicom.read_file(filename)
    niftiObject = dicomreaders.mosaic_to_nii(dicomObject)
    print(nib.aff2axcodes(niftiObject.affine))
    # temp_data = niftiObject.get_data()
    # correct_object = new_img_like(scratch_bold_ref, output_image_correct, copy_header=True)
    splitList=filename.split('/')
    # fullNiftiFilename="/".join(splitList[0:-1])+'/'+splitList[-1].split('.')[0]+'.nii.gz'
    fullNiftiFilename=os.path.join(tmp_folder, splitList[-1].split('.')[0]+'.nii.gz')
    print('fullNiftiFilename=',fullNiftiFilename)
    niftiObject.to_filename(fullNiftiFilename)
    return fullNiftiFilename

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
    scanNum = cfg.scanNum[0]
    runNum = cfg.runNum[0]

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
    "Allowed file types: %s" %allowedFileTypes)

    # obtain the path for the directory where the subject's dicoms live
    if cfg.isSynthetic:
        cfg.dicomDir = cfg.imgDir
    else:
        subj_imgDir = "{}.{}.{}".format(cfg.datestr, cfg.subjectName, cfg.subjectName)
        cfg.dicomDir = os.path.join(cfg.imgDir, subj_imgDir)
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

    ## select reference volume from previous day
    ## select template volume from today

    # skull strip and flirt align, save the transformation matrix from template to reference.
    # can I save the transformation matrix fitted by mcflirt and apply it on the data?
    Neurosketch_folder='/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/data/'
    AnatomicalFile='/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/0110171_neurosketch/data/nifti/0110171_neurosketch_anat_mprage_brain.nii.gz'
    day1RecognitionRunFile='/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/0110171_neurosketch021_0063_001'
    day2RecognitionRunFile='/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch/0110171_neurosketch021_0021_001'
    day2RealtimeFolder='/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/projects/tProject/dicomDir/example/neurosketch'
    # realtimeDicomFileNamePattern="{subject_ID}_neurosketch{runID}_{curr_TR}_001".format(subject_ID=0110171,runID=001-057,curr_TR=0001-0216)
    realtimeDicomFileNamePattern="{subject_ID}_neurosketch{runID}_{curr_TR}_001".format(subject_ID='0110171',runID='001',curr_TR='0001')
    # 0110171_neurosketch021_0063_001
    tmp_folder='/tmp/kp578/'
    if not os.path.isdir(tmp_folder): 
        os.mkdir(tmp_folder)
                                
    ref=AnatomicalFile
    # nii_ref=dicom2nii(ref)

    A = time.time()
    nii=dicom2nii(day1RecognitionRunFile)

    nii_aligned=nii[0:-7]+'_aligned.nii.gz'
    # command = f"mcflirt -in {nii} -reffile {nii_ref} -out {nii_aligned} -plots"
    command = f'flirt -in {nii} -ref {ref} -out {nii_aligned} -omat {tmp_folder}day1RecognitionRunFile_2_AnatomicalFile.mat' # this is not working due to skull existance
    print('Running ' + command)
    call(command, shell=True)

    B = time.time()
    print('time=',B-A) 



    num_total_TRs = 20  # number of TRs to use for example 1
    # “{subject_ID}_neurosketch{runID}_{curr_TR}_001”.format(subject_ID=0110171,runID=001-057,curr_TR=0001-0216)

    if cfg.isSynthetic:
        num_total_TRs = cfg.numSynthetic
    all_avg_activations = np.zeros((num_total_TRs, 1))
    # initialize runData
    runData = StructDict()
    cfg.nVox=147456
    runData.story_data = np.zeros((cfg.nVox,num_total_TRs))
    _=[]
    for this_TR in np.arange(num_total_TRs):
        # declare variables that are needed to use 'readRetryDicomFromFileInterface'
        timeout_file = 5 # small number because of demo, can increase for real-time

        # use 'getDicomFileName' from 'readDicom.py' to obtain the filename structure
        #   of the dicom data you want to get... which is useful considering how
        #   complicated these filenames can be!
        #   INPUT:
        #       [1] cfg (config parameters)
        #       [2] scanNum (scan number)
        #       [3] fileNum (TR number, which will reference the correct file)
        #   OUTPUT:
        #       [1] fullFileName (the filename of the dicom that should be grabbed)
        print(f'scanNum={scanNum} this_TR={this_TR} cfg.dicomDir={cfg.dicomDir} cfg.dicomNamePattern={cfg.dicomNamePattern}')
        fileName = getDicomFileName(cfg, scanNum, this_TR+1)

        # use 'readRetryDicomFromFileInterface' in 'readDicom.py' to wait for dicom
        #   files to come in (by using 'watchFile' in 'fileClient.py') and then
        #   reading the dicom file once it receives it detected having received it
        #   INPUT:
        #       [1] fileInterface (this will allow a script from the cloud to access files
        #               from the stimulus computer that receives dicoms from the Siemens
        #               console computer)
        #       [2] filename (for the dicom file we're watching for and want to load)
        #       [3] timeout (time spent waiting for a file before timing out)
        #   OUTPUT:
        #       [1] dicomData (with class 'pydicom.dataset.FileDataset')
        print("• use 'readRetryDicomFromFileInterface' to read dicom file for",
            "TR %d, %s" %(this_TR, fileName))
        dicomData = readRetryDicomFromFileInterface(fileInterface, fileName,
            timeout_file)

        if cfg.isSynthetic:
            niftiObject = convertDicomImgToNifti(dicomData)
        else:
            # use 'dicomreaders.mosaic_to_nii' to convert the dicom data into a nifti
            #   object. additional steps need to be taken to get the nifti object in
            #   the correct orientation, but we will ignore those steps here. refer to
            #   the 'advanced sample project' for more info about that
            print("| convert dicom data into a nifti object")
            niftiObject = dicomreaders.mosaic_to_nii(dicomData)



        ############################################################
        ############################################################
        ############################################################
        ################## Code for preprocessing###################
        ############################################################
        ############################################################
        ############################################################


        # save nifti to file
        # align realtime nifti to template file
        # apply mask
        # extract data from aligned nifti

        # take the average of all the activation values
        niftiData=niftiObject.get_data().reshape(-1)
        runData.story_data[:,this_TR] = niftiData
        _.append(niftiData)
        avg_niftiData = np.mean(niftiData)
        avg_niftiData = np.round(avg_niftiData,decimals=2)
        print("| average activation value for TR %d is %f" %(this_TR, avg_niftiData))

        # max_niftiData = np.amax(niftiObject.get_data())
        # print("| max activation value for TR %d is %d" %(this_TR, max_niftiData))

        
        def preprocessData(cfg,dataMatrix,previous_badVoxels=None):
            # steps: zscore///check bad voxels//remove signal average
            # remove bad voxels
            # bad voxel criteria: (1) if raw signal < 100 OR std is < 1E-3 ( I think we're going to set it equal to 0 anyway)

            t_end = np.shape(dataMatrix)[1]
            zscoredData = stats.zscore(dataMatrix,axis=1,ddof = 1)
            zscoredData = np.nan_to_num(zscoredData)
            # remove story TRs
            # remove story average
            std = np.std(dataMatrix,axis=1,ddof=1)
            non_changing_voxels = np.argwhere(std < 1E-3)
            low_value_voxels = np.argwhere(np.min(dataMatrix,axis=1) < 100)
            badVoxels = np.unique(np.concatenate((non_changing_voxels,low_value_voxels)))
            # now combine with previously made badvoxels
            if previous_badVoxels is not None:
                updated_badVoxels = np.unique(np.concatenate((previous_badVoxels,badVoxels)))
            else:
                updated_badVoxels = badVoxels
            signalAvg = getAvgSignal(cfg) 
            preprocessedData = zscoredData - signalAvg[:,0:t_end]
            return preprocessedData, updated_badVoxels
        def preprocessAndPredict(cfg,runData,TRindex_story):
            """Predict cheating vs. paranoid probability at given station"""
            # stationInd = np.argwhere(TRindex_story == cfg.last_tr_in_station.astype(int))[0][0]
            # stationKey = 'station' + str(stationInd)
            # prevStationKey = 'station' + str(stationInd - 1)
            # print('this station is %i' % stationInd)
            print('this story TR is %i' % TRindex_story)
            # indexing for data goes to +1 because we want the index to include the last station TR
            if stationInd == 0 or len(runData.badVoxels) == 0:
                runData.dataForClassification[stationKey],runData.badVoxels[stationKey] = preprocessData(cfg,runData.story_data[:,0:TRindex_story+1])
            else:
                runData.dataForClassification[stationKey],runData.badVoxels[stationKey] = preprocessData(cfg,runData.story_data[:,0:TRindex_story+1],runData.badVoxels[prevStationKey])
            loaded_model = loadClassifier(cfg,stationInd)
            this_station_TRs = np.array(cfg.stationsDict[stationInd])
            n_station_TRs = len(this_station_TRs)
            if len(runData.badVoxels[stationKey]) > 0:
                voxelsToExclude = runData.badVoxels[stationKey]
                print('excluding {0} voxels-- {1}'.format(len(voxelsToExclude),voxelsToExclude))
                runData.dataForClassification[stationKey][voxelsToExclude,:] = 0
            thisStationData = runData.dataForClassification[stationKey][:,this_station_TRs]
            dataForClassification_reshaped = np.reshape(thisStationData,(1,cfg.nVox*n_station_TRs))
            runData.cheating_probability[stationInd] = loaded_model.predict_proba(dataForClassification_reshaped)[0][1]

            a = np.load(cfg.station_stats)
            all_means = a['m']
            all_std = a['s']  
            runData.zTransferred[stationInd] = getTransferredZ(runData.cheating_probability[stationInd],stationInd,all_means,all_std)
            if runData.interpretation == 'C':
                runData.correct_prob[stationInd] = runData.zTransferred[stationInd]
            elif runData.interpretation == 'P':
                runData.correct_prob[stationInd] = 1 - runData.zTransferred[stationInd]
            return runData

        A = time.time()
        runData = preprocessAndPredict(cfg,runData,this_TR)
        B = time.time()
        print('preprocessAndPredict time: {:5f}'.format(B-A))

        ############################################################
        ############################################################
        ############################################################
        ################## Code for preprocessing###################
        ############################################################
        ############################################################
        ############################################################


        # use 'sendResultToWeb' from 'projectUtils.py' to send the result to the
        #   web browser to be plotted in the --Data Plots-- tab.
        print("| send result to the web, plotted in the 'Data Plots' tab")
        projUtils.sendResultToWeb(projectComm, runNum, int(this_TR), float(avg_niftiData))

        # save the activations value info into a vector that can be saved later
        all_avg_activations[this_TR] = avg_niftiData

    # create the full path filename of where we want to save the activation values vector
    #   we're going to save things as .txt and .mat files
    output_textFilename = '/tmp/kp578/cloud_directory/tmp/avg_activations.txt'
    output_matFilename = os.path.join('/tmp/kp578/cloud_directory/tmp/avg_activations.mat')

    # use 'putTextFile' from 'fileClient.py' to save the .txt file
    #   INPUT:
    #       [1] filename (full path!)
    #       [2] data (that you want to write into the file)
    print(""
    "-----------------------------------------------------------------------------\n"
    "• save activation value as a text file to tmp folder")
    fileInterface.putTextFile(output_textFilename,str(all_avg_activations))

    # use sio.save mat from scipy to save the matlab file
    print("• save activation value as a matlab file to tmp folder")
    sio.savemat(output_matFilename,{'value':all_avg_activations})

    print(""
    "-----------------------------------------------------------------------------\n"
    "REAL-TIME EXPERIMENT COMPLETE!")
    _=np.asarray(_)
    print('all data dimensions=',_.shape)

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
    cfg = loadConfigFile(defaultConfig)
    # print('args.config=',defaultConfig)
    print('cfg.subjectName=',cfg.subjectName)

    # obtain paths for important directories (e.g. location of dicom files)
    if cfg.imgDir is None:
        cfg.imgDir = os.path.join(currPath, 'dicomDir/example')
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


