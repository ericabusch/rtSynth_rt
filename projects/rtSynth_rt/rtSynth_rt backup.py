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

verbose = True

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
from rtCommon.utils import loadConfigFile
import rtCommon.clientInterface as clientInterface
from rtCommon.imageHandling import readRetryDicomFromFileInterface, getDicomFileName, convertDicomImgToNifti
sys.path.append('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/')

def gaussian(x, mu, sig):
    # mu and sig is determined before each neurofeedback session using 2 recognition runs.
    return round(1+18*(1 - np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))) # map from (0,1) -> [1,19]


# obtain the full path for the configuration toml file
# defaultConfig = os.path.join(currPath, 'conf/sample.toml')
defaultConfig = '/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/projects/rtSynth_rt/'+"sub001.ses2.toml"


def doRuns(cfg, fileInterface, subjInterface):
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
    # if verbose:
    #     print("Location of the subject's dicoms: \n%s\n" %cfg.dicomDir,
    #     "-----------------------------------------------------------------------------")

    # initialize a watch for the entire dicom folder (it doesn't look for a
    #   specific dicom) using the function 'initWatch' in 'fileClient.py'
    #   INPUT:
    #       [1] cfg.dicomDir (where the subject's dicom files live)
    #       [2] cfg.dicomNamePattern (the naming pattern of dicom files)
    #       [3] cfg.minExpectedDicomSize (a check on size to make sure we don't
    #               accidentally grab a dicom before it's fully acquired)
    if verbose:
        print("• initalize a watch for the dicoms using 'initWatch'")

    print(f"cfg.dicom_dir={cfg.dicom_dir}, cfg.dicomNamePattern={cfg.dicomNamePattern}, cfg.minExpectedDicomSize={cfg.minExpectedDicomSize}")
    fileInterface.initWatch(cfg.dicom_dir, cfg.dicomNamePattern,
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
    if verbose:
        print("• clear any pre-existing plot using 'sendResultToWeb'")
    subjInterface.sendClassificationResult(runNum, None, None)

    if verbose:
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
    
    tmp_dir=f"{cfg.tmp_folder}{time.time()}/" ; mkdir(tmp_dir)

    num_total_TRs = cfg.num_total_TRs  # number of TRs to use for example 1
    morphParams = np.zeros((num_total_TRs, 1))
    nift_data=[]
    for this_TR in np.arange(1,num_total_TRs):
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
        fileName = getDicomFileName(cfg, scanNum, this_TR)

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
        print(f'Processing TR {this_TR}')
        if verbose:
            print("• use 'readRetryDicomFromFileInterface' to read dicom file for",
                "TR %d, %s" %(this_TR, fileName))
        dicomData = readRetryDicomFromFileInterface(fileInterface, fileName, timeout_file)
        


        # use 'dicomreaders.mosaic_to_nii' to convert the dicom data into a nifti
        #   object. additional steps need to be taken to get the nifti object in
        #   the correct orientation, but we will ignore those steps here. refer to
        #   the 'advanced sample project' for more info about that
        if verbose:
            print("| convert dicom data into a nifti object")
        niftiObject = dicomreaders.mosaic_to_nii(dicomData)
        # print(f"niftiObject={niftiObject}")


        # save(f"{tmp_dir}niftiObject")
        niiFileName=f"{tmp_dir}{fileName.split('/')[-1].split('.')[0]}.nii"
        nib.save(niftiObject, niiFileName)  
        # align -in f"{tmp_dir}niftiObject" -ref cfg.templateFunctionalVolume_converted -out f"{tmp_dir}niftiObject"
        command = f"3dvolreg \
                -base {cfg.templateFunctionalVolume_converted} \
                -prefix  {niiFileName} \
                {niiFileName}"
        call(command,shell=True)
        niftiObject = nib.load(niiFileName)
        nift_data.append(niftiObject.get_fdata())
        

        # load f"{tmp_dir}niftiObject"
        # load cfg.chosenMask
        mask=nib.load(cfg.chosenMask).get_data()
        
        # load clf
        [mu,sig]=np.load(f"{cfg.feedback_dir}morphingTarget.npy")
        print(f"mu={mu},sig={sig}")

        # getting MorphingParameter: 
        # which clf to load? 
        # B evidence in BC/BD classifier for currt TR

        def classifierEvidence(clf,X,Y): # X shape is [trials,voxelNumber], Y is ['bed', 'bed'] for example # return a 1-d array of probability
            # This function get the data X and evidence object I want to know Y, and output the trained model evidence.
            targetID=[np.where((clf.classes_==i)==True)[0][0] for i in Y]
            Evidence=(np.sum(X*clf.coef_,axis=1)+clf.intercept_) if targetID[0]==1 else (1-(np.sum(X*clf.coef_,axis=1)+clf.intercept_))
            return np.asarray(Evidence)

        print(f"nift_data[-1].shape={nift_data[-1].shape}")
        print(f"mask.shape={mask.shape}")
        print(f"np.sum(mask)={np.sum(mask)}")
        X = nift_data[-1][mask==1]
        X = np.expand_dims(X, axis=0)
        print(f"X.shape={X.shape}")
        
        import joblib
        Y = ['chair'] * X.shape[0]
        # imcodeDict={
        # 'A': 'bed',
        # 'B': 'chair',
        # 'C': 'table',
        # 'D': 'bench'}
        BC_clf=joblib.load(cfg.usingModel_dir +'benchchair_chairtable.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
        BD_clf=joblib.load(cfg.usingModel_dir +'bedchair_chairbench.joblib') # These 4 clf are the same: bedbench_benchtable.joblib bedtable_tablebench.joblib benchchair_benchtable.joblib chairtable_tablebench.joblib
        BC_B_evidence = classifierEvidence(BC_clf,X,Y)[0]
        BD_B_evidence = classifierEvidence(BD_clf,X,Y)[0]
        print(f"BC_B_evidence={BC_B_evidence}")
        print(f"BD_B_evidence={BD_B_evidence}")
        print(f"(BC_B_evidence+BD_B_evidence)/2={(BC_B_evidence+BD_B_evidence)/2}")
        print(f"mu={mu}, sig={sig}")
        morphParam=gaussian((BC_B_evidence+BD_B_evidence)/2, mu, sig)
        print(f"morphParam={morphParam}")


        print("| morphParam for TR %d is %f" %(this_TR, morphParam))

        # use 'sendResultToWeb' from 'projectUtils.py' to send the result to the
        #   web browser to be plotted in the --Data Plots-- tab.
        if verbose:
            print("| send result to the web, plotted in the 'Data Plots' tab")
        subjInterface.sendClassificationResult(runNum, int(this_TR), morphParam)
        fileInterface.putTextFile("/tmp/test.txt",str(morphParam))

        # save the activations value info into a vector that can be saved later
        morphParams[this_TR] = morphParam

        
        time.sleep(1.5)

    # create the full path filename of where we want to save the activation values vector
    #   we're going to save things as .txt and .mat files
    output_textFilename = f'{cfg.feedback_dir}morphParam.txt'
    output_matFilename = os.path.join(f'{cfg.feedback_dir}morphParam.mat')

    # use 'putTextFile' from 'fileClient.py' to save the .txt file
    #   INPUT:
    #       [1] filename (full path!)
    #       [2] data (that you want to write into the file)
    if verbose:
        print(""
        "-----------------------------------------------------------------------------\n"
        "• save activation value as a text file to tmp folder")
    fileInterface.putTextFile(output_textFilename,str(morphParams))

    # use sio.save mat from scipy to save the matlab file
    if verbose:
        print("• save activation value as a matlab file to tmp folder")
    sio.savemat(output_matFilename,{'value':morphParam})

    if verbose:
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

    args = argParser.parse_args(argv)

    # Initialize the RPC connection to the projectInterface
    # This will give us a fileInterface for retrieving files and
    # a subjectInterface for giving feedback
    clientRPC = clientInterface.ClientRPC()
    fileInterface = clientRPC.fileInterface
    subjInterface = clientRPC.subjInterface


    # load the experiment configuration file
    # cfg = loadConfigFile(args.config)
    
    # config="rtSynth_rt.toml" #"sub001.ses1.toml"
    print(f"rtSynth_rt: args.config={args.config}")
    cfg = cfg_loading(args.config)


    # obtain paths for important directories (e.g. location of dicom files)
    if cfg.imgDir is None:
        cfg.imgDir = os.path.join(currPath, 'dicomDir')
    cfg.codeDir = currPath



    # now that we have the necessary variables, call the function 'doRuns' in order
    #   to actually start reading dicoms and doing your analyses of interest!
    #   INPUT:
    #       [1] cfg (configuration file with important variables)
    #       [2] fileInterface (this will allow a script from the cloud to access files
    #               from the stimulus computer that receives dicoms from the Siemens
    #               console computer)
    #       [3] subjInterface (to send/receive feedback to the subject in the scanner)
    doRuns(cfg, fileInterface, subjInterface)

    return 0


if __name__ == "__main__":
    """
    If 'sample.py' is invoked as a program, then actually go through all of the portions
    of this script. This statement is not satisfied if functions are called from another
    script using "from sample.py import FUNCTION"
    """
    main()
    sys.exit(0)
