#!/usr/bin/env bash
# Input python command to be submitted as a job
#SBATCH --output=logs/makeGreyMatterMask-%j.out
#SBATCH --job-name makeGreyMatterMask
#SBATCH --partition=short,scavenge,day
#SBATCH --time=2:00:00
#SBATCH --mem=10000
module load AFNI
module load FreeSurfer/6.0.0


# 让当前的 makeGreyMatterMask.sh 在这个folder GreyMatterMask里面运行
subject=$1
scan_asTemplate=$2

anatPath=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/${subject}/ses1/anat/
subjectFolder=${anatPath}freesurfer/${subject}/

GreyMatterMask=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/${subject}/ses1/GreyMatterMask/
mkdir -p ${GreyMatterMask}
cd ${GreyMatterMask}

#复制SUMA的结果到当前的folder
SUMA_dir=${subjectFolder}/SUMA/
SUMA_result=${SUMA_dir}${subject}_SurfVol+orig 
3dcopy ${SUMA_result} .  #copy 1206161_SurfVol+orig.BRIK  1206161_SurfVol+orig.HEAD

#做左半球的皮层，将数据从surface domain 投射到AFNI volume domain ；也就是lh_gm+orig.BRIK以及lh_gm+orig.HEAD
3dSurf2Vol -spec ${SUMA_dir}${subject}_lh.spec -surf_A smoothwm -surf_B pial -sv ${subject}_SurfVol+orig -grid_parent ${subject}_SurfVol+orig -map_func mask2 -f_steps 50 -prefix lh_gm 
#做右半球的皮层，将数据从surface domain 投射到AFNI volume domain ；也就是rh_gm+orig.BRIK以及rh_gm+orig.HEAD
3dSurf2Vol -spec ${SUMA_dir}${subject}_rh.spec -surf_A smoothwm -surf_B pial -sv ${subject}_SurfVol+orig -grid_parent ${subject}_SurfVol+orig -map_func mask2 -f_steps 50 -prefix rh_gm 

#将左右半球的数据加和到一起，a是左半球，b是右半球，进行的运算是or运算，也就是 or（a==1，b==1），prefix 给的是结果的命名方式;制造出了gm+orig.HEAD 以及gm+orig.BRIK
3dcalc -a lh_gm+orig -b rh_gm+orig -expr 'or(equals(a,1),equals(b,1))' -prefix gm 
'''
# 当前文件夹内容是：
# 1206161_SurfVol+orig.BRIK  gm+orig.HEAD     rh_gm+orig.BRIK
# 1206161_SurfVol+orig.HEAD  lh_gm+orig.BRIK  rh_gm+orig.HEAD
# gm+orig.BRIK		   lh_gm+orig.HEAD
'''
#去除T1的颅骨
3dSkullStrip -input ../anat/T1.nii -prefix anat_stripped+orig

#去除皮层的颅骨
3dSkullStrip -input ${subject}_SurfVol+orig -prefix ${subject}_SurfVol_stripped 

# 把T1和皮层对准中心,将dset转移到base的中心，child是使用dset到base的转移矩阵将child的数据转移到base的中心。在这个例子中是将灰质mask以及‘未经SUMA处理过的，只被freesurfer处理过的灰质mask’两个数据库转移到T1的中心
\@Align_Centers -base anat_stripped+orig -dset ${subject}_SurfVol_stripped+orig -child gm+orig ${subject}_SurfVol+orig
# 这一行code会产生一系列的shft的数据

#将 去除掉颅骨的皮层数据 转移到 去除掉颅骨的T1数据 的空间内，可能是转移方向，也可能是upsample/downsample
3dresample -master anat_stripped+orig -prefix ${subject}_SurfVol_stripped_shft_resamp+orig -inset ${subject}_SurfVol_stripped_shft+orig
# 这一行code会产生 resamp 的数据

#对两个数据进行align，转换一个数据到另一个数据的空间内，具体的，将灰质皮层的mask的结果与去除过颅骨的T1数据
# -dset1 数据集1 -dset2 数据集2 -dset2to1 想要的方向是数据集2到1 -dset1_strip 数据集1是否需要用任何去除颅骨的算法 -dset2_strip 数据集2是否需要用任何去除颅骨的算法 
# -Allineate_opts 记性align用的参数，默认的是"-weight_frac 1.0 -maxrot 6 -maxshf 10 -VERB -warp aff "，现在采用的是"-VERB -warp aff "也就是默认的
# -source_automask 这是 Allineate_opts的一个选项，含义不明
# -child_epi gm_shft+orig：这是把dset2到1的变化应用到child_epi上面
align_epi_anat.py -dset1 anat_stripped+orig -dset2 ${subject}_SurfVol_stripped_shft_resamp+orig -dset2to1 -dset1_strip None -dset2_strip None -Allineate_opts "-VERB -warp aff " -source_automask -child_epi gm_shft+orig 

#使用2.0mm FWHM Gaussian blur到灰质mask上面
3dmerge -1blur_fwhm 2.0 -prefix gm_shft_aligned_smooth gm_shft_al+orig 

# 将已经搞好的在T1 space的灰质mask转移到functional space里面去
# processedEPI=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${subject}/neurosketch_recognition_run_1_bet.nii.gz # processedEPI+orig 是一个处理好的functional 数据
processedEPI=/gpfs/milgram/project/turk-browne/projects/rtSynth_rt/subjects/${subject}/ses1/recognition/templateFunctionalVolume.nii 
processedEPI_bet=../anat/functional_bet.nii
if test -f "$processedEPI"; then
    echo "$processedEPI exists."
else
    python -u /gpfs/milgram/project/turk-browne/projects/rtSynth_rt/expScripts/recognition/8runRecgnitionModelTraining.py -c ${subject}.ses1.toml --scan_asTemplate ${scan_asTemplate} --preprocessOnly
fi

bet ${processedEPI} ${processedEPI_bet}
3dresample -master ${processedEPI_bet} -prefix gm_shft_aligned_smooth_resamp+orig -input gm_shft_aligned_smooth+orig
3dresample -input gm_shft_aligned_smooth_resamp+orig -prefix gm_resamp.nii.gz # 将afni的数据转化成为nifti的数据格式

# 最终的functional的灰质mask是 gm_shft_aligned_smooth_resamp+orig



3dresample -master ../anat/T1.nii -prefix gm_anat+orig -input gm_shft_aligned_smooth+orig
3dresample -input gm_anat+orig -prefix gm_anat.nii.gz #gm_anat.nii.gz 是在anat空间里面的gm mask
# 将T1.nii去掉颅骨
bet ../anat/T1.nii ../anat/T1_bet.nii
# 制造T1.nii到processedEPI的投射
# flirt -ref ${processedEPI_bet} -in ../anat/T1_bet.nii -out ../anat/T1inFunc.nii -omat ../anat/T1inFunc.mat -dof 6
TEMPLATE_bet=$processedEPI_bet
ANAT_bet=../anat/T1_bet.nii
ANAT2FUNC=anat2func.mat
ANATinFUNC=ANATinFUNC.nii.gz
flirt -ref $TEMPLATE_bet -in $ANAT_bet -omat $ANAT2FUNC -out $ANATinFUNC -dof 6 # flirt -ref $TEMPLATE_bet -in $ANAT_bet -omat $ANAT2FUNC -out $ANATinFUNC -dof 6
# fslview_deprecated ../anat/T1inFunc.nii ${processedEPI_bet}
# 用T1.nii到processedEPI的投射 将gm转移到processedEPI空间里面
flirt -ref ${TEMPLATE_bet} -in gm_anat.nii.gz -out ../anat/gm_func.nii.gz -applyxfm -init $ANAT2FUNC
# fslview_deprecated ${processedEPI_bet} ../anat/gm_func.nii.gz
# 查看效果，产生的gm是否在functional space里面。
# fslview_deprecated ${functionGreyMatter} ${processedEPI}
# fslview_deprecated gm.nii.gz ${processedEPI}