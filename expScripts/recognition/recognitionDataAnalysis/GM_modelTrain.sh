#!/usr/bin/env bash
#SBATCH --output=logs/GM_modelTrain-%j.out
#SBATCH -p day
#SBATCH -t 24:00:00
#SBATCH --mem 20GB
#SBATCH -n 1
module load AFNI
module load FreeSurfer/6.0.0
module load FSL
. ${FSLDIR}/etc/fslconf/fsl.sh
set -e

code_dir=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/
raw_dir=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/recognitionDataAnalysis/raw/
cd ${code_dir}

sub=$1 # rtSynth_sub001 rtSynth_sub001_ses5 rtSynth_sub001 rtSynth_sub002_ses1
scan_asTemplate=$2

python -u -c "from GM_modelTrain_functions import _split ; _split('${sub}')"
source ${code_dir}${sub}_subjectName.txt # so you have subjectName and ses
echo subjactName=${subjectName} ses=${ses}

# 下载dcm数据。并且在raw_dir 里面产生{sub}_run_name.txt 用于储存每一个run分别对应什么。举例来说比如 尾数为8的代表T1 数据。
bash ${code_dir}fetchXNAT.sh ${sub}

# 通过对{sub}_run_name.txt的处理获得第二个 ABCD_T1w_MPR_vNav   usable 前面的数字
python -u -c "from GM_modelTrain_functions import find_ABCD_T1w_MPR_vNav; find_ABCD_T1w_MPR_vNav('$sub')"
source ${raw_dir}${sub}_ABCD_T1w_MPR_vNav.txt # 加载 T1_ID 由find_ABCD_T1w_MPR_vNav 产生
echo T1_ID=${T1_ID} T2_ID=${T2_ID}

# 等到zip file完成
python -u -c "from GM_modelTrain_functions import wait; wait('${raw_dir}${sub}.zip')"

# unzip
cd ${raw_dir}
unzip ${sub}.zip

# 把dcm变成nii
cd ${code_dir}
bash ${code_dir}change2nifti.sh ${sub}

# 根据找到的第二个T1 图像，移动到subject folder里面对应的ses的anat folder。
python -u -c "from GM_modelTrain_functions import find_T1_in_niiFolder ; find_T1_in_niiFolder(${T1_ID},${T2_ID},'${sub}')"

# 运行freesurfer
cd /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/
sbatch reconsurf.sh ${subjectName}

# 等待 freesurfer 完成
anatPath=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/${subjectName}/ses1/anat/
cd ${code_dir}
echo python -u -c "from GM_modelTrain_functions import wait; wait('${anatPath}done_${subjectName}.txt')"
python -u -c "from GM_modelTrain_functions import wait; wait('${anatPath}done_${subjectName}.txt')"

# SUMA_Make_Spec_FS.sh
cd /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/
sbatch SUMA_Make_Spec_FS.sh ${subjectName}

# 等待 SUMA_Make_Spec_FS 完成
cd ${code_dir}
echo python -u -c "from GM_modelTrain_functions import wait; wait('${anatPath}SUMAdone_${subjectName}.txt')"
python -u -c "from GM_modelTrain_functions import wait; wait('${anatPath}SUMAdone_${subjectName}.txt')"

# 获得mask
cd /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/
sbatch makeGreyMatterMask.sh ${subjectName} ${scan_asTemplate}

# 产生的mask 类似 /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/sub001/ses1/anat/gm_func.nii.gz
cd ${code_dir}
echo python -u -c "from GM_modelTrain_functions import wait; wait('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/${subjectName}/ses${ses}/anat/gm_func.nii.gz')"
python -u -c "from GM_modelTrain_functions import wait; wait('/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/${subjectName}/ses${ses}/anat/gm_func.nii.gz')"

# 下一步是 greedy 以及 训练模型
cd /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/
echo python -u expScripts/recognition/8runRecgnitionModelTraining.py -c ${subjectName}.ses${ses}.toml --scan_asTemplate ${scan_asTemplate}
python -u expScripts/recognition/8runRecgnitionModelTraining.py -c ${subjectName}.ses${ses}.toml --scan_asTemplate ${scan_asTemplate} --skipPre
