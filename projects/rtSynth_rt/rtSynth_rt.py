'''
steps:
    conver the new dicom to nii
    align the nii to cfg.templateFunctionalVolume_converted
    apply mask 
    load clf
    get morphing parameter
'''

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

verbose = False
useInitWatch = True

if verbose:
    # print a short introduction on the internet window
    print(""
        "-----------------------------------------------------------------------------\n"
        "The purpose of this sample project is to demonstrate different ways you can\n"
        "implement functions, structures, etc. that we have developed for your use.\n"
        "You will find some comments printed on this html browser. However, if you want\n"
        "more information about how things work please take a look at ‘sample.py’.\n"
        "Good luck!\n"
        "-----------------------------------------------------------------------------")

# import important modules
import os,time
import sys
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/')
import argparse
import warnings
import numpy as np
import nibabel as nib
import scipy.io as sio
from rtCommon.cfg_loading import mkdir,cfg_loading
from subprocess import call
import joblib
from scipy.stats import zscore
if verbose:
    print(''
        '|||||||||||||||||||||||||||| IGNORE THIS WARNING ||||||||||||||||||||||||||||')
with warnings.catch_warnings():
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning)
    from nibabel.nicom import dicomreaders

if verbose:
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
from rtCommon.utils import loadConfigFile, stringPartialFormat
from rtCommon.clientInterface import ClientInterface
from rtCommon.imageHandling import readRetryDicomFromDataInterface, convertDicomImgToNifti
from rtCommon.dataInterface import DataInterface #added by QL
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/')
from recognition_dataAnalysisFunctions import normalize,classifierEvidence

# def classifierEvidence(clf,X,Y): # X shape is [trials,voxelNumber], Y is ['bed', 'bed'] for example # return a 1-d array of probability
#     # This function get the data X and evidence object I want to know Y, and output the trained model evidence.
#     targetID=[np.where((clf.classes_==i)==True)[0][0] for i in Y]
#     # Evidence=(np.sum(X*clf.coef_,axis=1)+clf.intercept_) if targetID[0]==1 else (1-(np.sum(X*clf.coef_,axis=1)+clf.intercept_))
#     Evidence=(X@clf.coef_.T+clf.intercept_) if targetID[0]==1 else (1-(X@clf.coef_.T+clf.intercept_))
#     Evidence = 1/(1+np.exp(-Evidence))
#     # Evidence = sigmoid(Evidence)
#     return np.asarray(Evidence)

# def classifierEvidence(clf,X,Y):
#     ID=np.where((clf.classes_==Y[0])*1==1)
#     p = clf.predict_proba(X)[:,ID]
#     BX=np.log(p/(1-p))
#     return BX

# def classifierEvidence(clf,X,Y):
#     ID=np.where((clf.classes_==Y[0])*1==1)[0][0]
#     Evidence=(X@clf.coef_.T+clf.intercept_) if ID==1 else (-(X@clf.coef_.T+clf.intercept_))
#     return np.asarray(Evidence)

sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/')

def gaussian(x, mu, sig):
    # mu and sig is determined before each neurofeedback session using 2 recognition runs.
    return round(1+18*(1 - np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))) # map from (0,1) -> [1,19]



# obtain the full path for the configuration toml file
# defaultConfig = os.path.join(currPath, 'conf/sample.toml')
defaultConfig = '/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/projects/rtSynth_rt/'+"sub001.ses3.toml"


def doRuns(cfg, dataInterface, subjInterface, webInterface):
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

    print(f"Doing run {runNum}, scan {scanNum}")

    """
    Before we get ahead of ourselves, we need to make sure that the necessary file
        types are allowed (meaning, we are able to read them in)... in this example,
        at the very least we need to have access to dicom and txt file types.
    use the function 'allowedFileTypes' in 'fileClient.py' to check this!
    If allowedTypes doesn't include the file types we need to use then the 
        file service (scannerDataService) running at the control room computer will
        need to be restarted with the correct list of allowed types provided.

    INPUT: None
    OUTPUT:
          [1] allowedFileTypes (list of allowed file types)
    """
    allowedFileTypes = dataInterface.getAllowedFileTypes()
    if verbose:
        print(""
        "-----------------------------------------------------------------------------\n"
        "Before continuing, we need to make sure that dicoms are allowed. To verify\n"
        "this, use the 'allowedFileTypes'.\n"
        "Allowed file types: %s" %allowedFileTypes)

    # obtain the path for the directory where the subject's dicoms live
    # if cfg.isSynthetic:
    #     cfg.dicomDir = cfg.imgDir
    # else:
    #     subj_imgDir = "{}.{}.{}".format(cfg.datestr, cfg.subjectName, cfg.subjectName)
    #     cfg.dicomDir = os.path.join(cfg.imgDir, subj_imgDir)
    if verbose:
        print("Location of the subject's dicoms: \n" + cfg.dicomDir + "\n"
        "-----------------------------------------------------------------------------")

    #  If a dicomNamePattern is supplied in the config file, such as
    #  "001_{SCAN:06d}_{TR:06d}.dcm", then call stringPartialFormat() to 
    #  set the SCAN number for the series of Dicoms we will be streaming.
    dicomScanNamePattern = stringPartialFormat(cfg.dicomNamePattern, 'SCAN', scanNum)
    print(f"dicomScanNamePattern={dicomScanNamePattern}")

    """
    There are several ways to receive Dicom data from the control room computer:
    1. Using `initWatch()` and 'watchFile()` commands of dataInterface or the
        helper function `readRetryDicomFromDataInterface()` which calls watchFile()
        internally.
    2. Using the streaming functions with `initScannerStream()` and `getImageData(stream)`
        which are also part of the dataInterface.
    """
    if useInitWatch is True:
        """
        Initialize a watch for the entire dicom folder using the function 'initWatch'
        of the dataInterface. (Later we will use watchFile() to look for a specific dicom)
        INPUT:
            [1] cfg.dicomDir (where the subject's dicom files live)
            [2] cfg.dicomNamePattern (the naming pattern of dicom files)
            [3] cfg.minExpectedDicomSize (a check on size to make sure we don't
                    accidentally grab a dicom before it's fully acquired)
        """
        if verbose:
            print("• initalize a watch for the dicoms using 'initWatch'")
            print(f"cfg.dicom_dir={cfg.dicom_dir}, cfg.dicomNamePattern={cfg.dicomNamePattern}, \
                cfg.minExpectedDicomSize={cfg.minExpectedDicomSize}")
        dataInterface.initWatch(cfg.dicomDir, dicomScanNamePattern, cfg.minExpectedDicomSize)

    else:  # use Stream functions
        """
        Initialize a Dicom stream by indicating the directory and dicom file pattern that
        will be streamed.

        INPUTs to initScannerStream():
            [1] cfg.dicomDir (where the subject's dicom files live)
            [2] dicomScanNamePattern (the naming pattern of dicom files)
            [3] cfg.minExpectedDicomSize (a check on size to make sure we don't
                    accidentally grab a dicom before it's fully acquired)
        """
        if verbose:
            print(f"cfg.dicomDir={cfg.dicomDir}, dicomScanNamePattern={dicomScanNamePattern}, cfg.minExpectedDicomSize={cfg.minExpectedDicomSize})")
            print(f"cfg.dicom_dir={cfg.dicom_dir}, cfg.dicomNamePattern={cfg.dicomNamePattern}, \
                cfg.minExpectedDicomSize={cfg.minExpectedDicomSize}")
        streamId = dataInterface.initScannerStream(cfg.dicomDir, 
                                                dicomScanNamePattern,
                                                cfg.minExpectedDicomSize)


    """
    We will use the function plotDataPoint in webInterface whenever we
      want to send values to the web browser so that they can be plotted in the
      --Data Plots-- tab. 
    However at the start of a run we will want to clear the plot, and we can use
    clearRunPlot(runId), or clearAllPlots() also in the webInterface object.
    """
    if verbose:
        print("• clear any pre-existing plot for this run using 'clearRunPlot(runNum)'")
    webInterface.clearRunPlot(runNum)

    if verbose:
        print(""
        "-----------------------------------------------------------------------------\n"
        "In this sample project, we will retrieve the dicom file for a given TR and\n"
        "then convert the dicom file to a nifti object. **IMPORTANT: In this sample\n"
        "we won't care about the exact location of voxel data (we're only going to\n"
        "indiscriminately get the average activation value for all voxels). This\n"
        "actually isn't something you want to actually do but we'll go through the\n"
        "to get the data in the appropriate nifti format in the advanced sample\n"
        "project (amygActivation).** We are doing things in this way because it is the simplest way\n"
        "we can highlight the functionality of rt-cloud, which is the purpose of\n"
        "this sample project.\n"
        ".............................................................................\n"
        "NOTE: We will use the function readRetryDicomFromDataInterface() to retrieve\n"
        "specific dicom files from the subject's dicom folder. This function calls\n"
        "'dataInterface.watchFile' to look for the next dicom from the scanner.\n"
        "Since we're using previously collected dicom data, this functionality is\n"
        "not particularly relevant for this sample project but it is very important\n"
        "when running real-time experiments.\n"
        "-----------------------------------------------------------------------------\n")

    tmp_dir=f"{cfg.tmp_folder}{time.time()}/" ; mkdir(tmp_dir)
    mask=np.load(cfg.chosenMask)

    # load clf
    [mu,sig]=np.load(f"{cfg.feedback_dir}morphingTarget.npy")
    print(f"mu={mu},sig={sig}")
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    # getting MorphingParameter: 
    # which clf to load? 
    # B evidence in BC/BD classifier for currt TR



    BC_clf=joblib.load(cfg.usingModel_dir +'benchchair_chairtable.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
    BD_clf=joblib.load(cfg.usingModel_dir +'bedchair_chairbench.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib

    # where the morphParams are saved
    output_textFilename = f'{cfg.feedback_dir}morphParam_{scanNum}.txt'
    output_matFilename = os.path.join(f'{cfg.feedback_dir}morphParam_{scanNum}.mat')

    num_total_TRs = cfg.num_total_TRs  # number of TRs to use for example 1
    morphParams = np.zeros((num_total_TRs, 1))
    B_evidences=[]
    maskedData=0
    for this_TR in np.arange(1,num_total_TRs):
        # declare variables that are needed to use 'readRetryDicomFromFileInterface'
        timeout_file = 5 # small number because of demo, can increase for real-time
        dicomFilename = dicomScanNamePattern.format(TR=this_TR)

        if useInitWatch is True:
            """
            Use 'readRetryDicomFromDataInterface' in 'imageHandling.py' to wait for dicom
                files to be written by the scanner (uses 'watchFile' internally) and then
                reading the dicom file once it is available.
            INPUT:
                [1] dataInterface (allows a cloud script to access files from the
                    control room computer)
                [2] filename (the dicom file we're watching for and want to load)
                [3] timeout (time spent waiting for a file before timing out)
            OUTPUT:
                [1] dicomData (with class 'pydicom.dataset.FileDataset')
            """
            print(f'Processing TR {this_TR}')
            if verbose:
                print("• use 'readRetryDicomFromDataInterface' to read dicom file for",
                    "TR %d, %s" %(this_TR, dicomFilename))
            dicomData = readRetryDicomFromDataInterface(dataInterface, dicomFilename,
                timeout_file)  

        else:  # use Stream functions
            """
            Use dataInterface.getImageData(streamId) to query a stream, waiting for a 
                dicom file to be written by the scanner and then reading the dicom file
                once it is available.
            INPUT:
                [1] dataInterface (allows a cloud script to access files from the
                    control room computer)
                [2] streamId - from initScannerStream() called above
                [3] TR number - the image volume number to retrieve
                [3] timeout (time spent waiting for a file before timing out)
            OUTPUT:
                [1] dicomData (with class 'pydicom.dataset.FileDataset')
            """
            print(f'Processing TR {this_TR}')
            if verbose:
                print("• use dataInterface.getImageData() to read dicom file for"
                    "TR %d, %s" %(this_TR, dicomFilename))
            dicomData = dataInterface.getImageData(streamId, int(this_TR), timeout_file)

        if dicomData is None:
            print('Error: getImageData returned None')
            return
   
        dicomData.convert_pixel_data()

        # use 'dicomreaders.mosaic_to_nii' to convert the dicom data into a nifti
        #   object. additional steps need to be taken to get the nifti object in
        #   the correct orientation, but we will ignore those steps here. refer to
        #   the 'advanced sample project' for more info about that
        if verbose:
            print("| convert dicom data into a nifti object")
        niftiObject = dicomreaders.mosaic_to_nii(dicomData)
        # print(f"niftiObject={niftiObject}")

        # save(f"{tmp_dir}niftiObject")
        # niiFileName=f"{tmp_dir}{fileName.split('/')[-1].split('.')[0]}.nii"
        niiFileName= tmp_dir+cfg.dicomNamePattern.format(SCAN=scanNum,TR=this_TR).split('.')[0] + ".nii"
        print(f"niiFileName={niiFileName}")
        nib.save(niftiObject, niiFileName)  
        # align -in f"{tmp_dir}niftiObject" -ref cfg.templateFunctionalVolume_converted -out f"{tmp_dir}niftiObject"
        command = f"3dvolreg \
                -base {cfg.templateFunctionalVolume_converted} \
                -prefix  {niiFileName} \
                {niiFileName}"
        call(command,shell=True)
        niftiObject = nib.load(niiFileName)
        nift_data = niftiObject.get_fdata()
        
        curr_volume = np.expand_dims(nift_data[mask==1], axis=0)
        maskedData=curr_volume if this_TR==1 else np.concatenate((maskedData,curr_volume),axis=0)
        _maskedData = normalize(maskedData)

        print(f"_maskedData.shape={_maskedData.shape}")
        # print(f"X.shape={X.shape}")
        X = np.expand_dims(_maskedData[-1], axis=0)
        # print(f"X.shape={X.shape}")
        # print(f"X={X}")
        
        Y = 'chair'
        # imcodeDict={
        # 'A': 'bed',
        # 'B': 'chair',
        # 'C': 'table',
        # 'D': 'bench'}
        print(f"classifierEvidence(BC_clf,X,Y)={classifierEvidence(BC_clf,X,Y)}")
        print(f"classifierEvidence(BD_clf,X,Y)={classifierEvidence(BD_clf,X,Y)}")
        BC_B_evidence = classifierEvidence(BC_clf,X,Y)[0]
        BD_B_evidence = classifierEvidence(BD_clf,X,Y)[0]
        print(f"BC_B_evidence={BC_B_evidence}")
        print(f"BD_B_evidence={BD_B_evidence}")
        B_evidence = float((BC_B_evidence+BD_B_evidence)/2)
        print(f"B_evidence={B_evidence}")
        print(f"mu={mu}, sig={sig}")
        morphParam=int(gaussian(B_evidence, mu, sig))
        B_evidences.append(B_evidence)
        print(f"morphParam={morphParam}")


        print("| morphParam for TR %d is %f" %(this_TR, morphParam))

        # use 'sendResultToWeb' from 'projectUtils.py' to send the result to the
        #   web browser to be plotted in the --Data Plots-- tab.
        

        if verbose:
            print("| send result to the presentation computer for provide subject feedback")
        subjInterface.setResult(runNum, int(this_TR), morphParam)

        if verbose:
            print("| send result to the web, plotted in the 'Data Plots' tab")
        webInterface.plotDataPoint(runNum, int(this_TR), B_evidence)

        # save the activations value info into a vector that can be saved later
        morphParams[this_TR] = morphParam

        dataInterface.putFile(output_textFilename,str(morphParams))
        np.save(f'{cfg.feedback_dir}B_evidences_{scanNum}',B_evidences)
        

        
        # time.sleep(1.5)

    # create the full path filename of where we want to save the activation values vector
    #   we're going to save things as .txt and .mat files
    

    # use 'putTextFile' from 'fileClient.py' to save the .txt file
    #   INPUT:
    #       [1] filename (full path!)
    #       [2] data (that you want to write into the file)
    if verbose:
        print(""
        "-----------------------------------------------------------------------------\n"
        "• save activation value as a text file to tmp folder")
    dataInterface.putFile(output_textFilename,str(morphParams))

    # use sio.save mat from scipy to save the matlab file
    if verbose:
        print("• save activation value as a matlab file to tmp folder")
    sio.savemat(output_matFilename,{'value':morphParams})

    if verbose:
        print(""
        "-----------------------------------------------------------------------------\n"
        "REAL-TIME EXPERIMENT COMPLETE!")

    return


def main(argv=None):
    global verbose, useInitWatch
    """
    This is the main function that is called when you run 'sample.py'.

    Here, you will load the configuration settings specified in the toml configuration
    file, initiate the clientInterface for communication with the projectServer (via
    its sub-interfaces: dataInterface, subjInterface, and webInterface). Ant then call
    the function 'doRuns' to actually start doing the experiment.
    """

    # Some generally recommended arguments to parse for all experiment scripts
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                           help='experiment config file (.json or .toml)')
    argParser.add_argument('--runs', '-r', default=None, type=str,
                           help='Comma separated list of run numbers')
    argParser.add_argument('--scans', '-s', default=None, type=str,
                           help='Comma separated list of scan number')
    argParser.add_argument('--yesToPrompts', '-y', default=False, action='store_true',
                           help='automatically answer tyes to any prompts')

    # Some additional parameters only used for this sample project
    argParser.add_argument('--useInitWatch', '-w', default=False, action='store_true',
                           help='use initWatch() functions instead of stream functions')
    argParser.add_argument('--Verbose', '-v', default=False, action='store_true',
                           help='print verbose output')

    args = argParser.parse_args(argv)

    useInitWatch = args.useInitWatch
    verbose = args.Verbose

    # load the experiment configuration file
    print(f"rtSynth_rt: args.config={args.config}")
    cfg = cfg_loading(args.config)


    # override config file run and scan values if specified
    if args.runs is not None:
        print("runs: ", args.runs)
        cfg.runNum = [int(x) for x in args.runs.split(',')]
    if args.scans is not None:
        print("scans: ", args.scans)
        cfg.ScanNum = [int(x) for x in args.scans.split(',')]

    # Initialize the RPC connection to the projectInterface.
    # This will give us a dataInterface for retrieving files,
    # a subjectInterface for giving feedback, and a webInterface
    # for updating what is displayed on the experimenter's webpage.
    clientInterfaces = ClientInterface(yesToPrompts=args.yesToPrompts)
    #dataInterface = clientInterfaces.dataInterface
    subjInterface = clientInterfaces.subjInterface
    webInterface  = clientInterfaces.webInterface

    ## Added by QL
    allowedDirs = ['*'] #['/gpfs/milgram/pi/turk-browne/projects/rt-cloud/projects/sample/dicomDir/20190219.0219191_faceMatching.0219191_faceMatching','/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/sample', '/gpfs/milgram/project/turk-browne/projects/rt-cloud/projects/sample/dicomDir']
    allowedFileTypes = ['*'] #['.txt', '.dcm']
    dataInterface = DataInterface(dataRemote=False,allowedDirs=allowedDirs,allowedFileTypes=allowedFileTypes) # Create an instance of local datainterface

    # Also try the placeholder for bidsInterface (an upcoming feature)
    bidsInterface = clientInterfaces.bidsInterface
    res = bidsInterface.echo("test")
    print(res)

    # obtain paths for important directories (e.g. location of dicom files)
    # if cfg.imgDir is None:
    #     cfg.imgDir = os.path.join(currPath, 'dicomDir')
    # cfg.codeDir = currPath

    # now that we have the necessary variables, call the function 'doRuns' in order
    #   to actually start reading dicoms and doing your analyses of interest!
    #   INPUT:
    #       [1] cfg (configuration file with important variables)
    #       [2] dataInterface (this will allow a script from the cloud to access files
    #            from the stimulus computer that receives dicoms from the Siemens
    #            console computer)
    #       [3] subjInterface - this allows sending feedback (e.g. classification results)
    #            to a subjectService running on the presentation computer to provide
    #            feedback to the subject (and optionally get their response).
    #       [4] webInterface - this allows updating information on the experimenter webpage.
    #            For example to plot data points, or update status messages.
    doRuns(cfg, dataInterface, subjInterface, webInterface)
    return 0


if __name__ == "__main__":
    """
    If 'sample.py' is invoked as a program, then actually go through all of the portions
    of this script. This statement is not satisfied if functions are called from another
    script using "from sample.py import FUNCTION"
    """
    main()
    sys.exit(0)


# def monitor(scnNum,config="sub001.ses3.toml"):
#     cfg = cfg_loading(config)
#     [mu,sig]=np.load(f"{cfg.feedback_dir}morphingTarget.npy")
# #     sig=0.5
#     B_evidences = np.load(f'{cfg.feedback_dir}B_evidences_{scnNum}.npy')
#     plt.plot(B_evidences)
#     plt.plot(np.arange(0,150),150*[mu])
#     plt.plot(np.arange(0,150),150*[mu+3*sig])
#     plt.plot(np.arange(0,150),150*[mu-3*sig])
#     print(f"mu={mu},sig={sig}")

#     _=plt.figure()
#     morphParam=[int(gaussian(B_evidence, mu, sig)) for B_evidence in B_evidences]
#     plt.plot(morphParam)
# monitor(2,config="sub001.ses2.toml")