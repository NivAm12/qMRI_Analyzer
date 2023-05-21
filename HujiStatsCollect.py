import os.path
from os import path
import os
import glob
import pandas as pd
import numpy as np
from re import search
import nibabel as nib
import subprocess

count_not_miss = 0
count_miss = 0
count = 0
#MIDGRAY
fsMainDir = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/freesurfer_subjects/'
##define parameter
outDir = '/ems/elsc-labs/mezer-a/shai.berman/Documents/Code/testing_pipelines/brainstem/FSstats'
paramId = 1

#t1stat = os.path.relpath('FSstats_midgray_T1.mat', statDir)
if paramId==1:
    statFile = os.path.relpath('FSstats_midgray_T1.mat',outDir)#Before we started revisiting the code in December 2019
                                                # it was: '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/doc/Claustrum/FSstats.mat';
    parcName = 'midgray_T1'
if paramId==2:
    statFile = os.path.relpath('FSstats_midgray_MTV.mat',outDir)
    parcName = 'midgray_MTV'
if paramId==3:
    statFile = os.path.relpath('FSstats_midgray_T2s.mat',outDir)
    parcName = 'midgray_R2star'
if paramId==4:
    statFile = os.path.relpath('FSstats_midgray_MT.mat',outDir)
    parcName = 'midgray_MT'



analysisDir = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'

# description: This function is the initial preprocessing of the subjects that will
# be included in the data analysis
# the subjects removed have technical problems in their initial values and analysis
# return value: numpy array shape (number of subjects after preprocessing,)



def HUJI_subjects_preprocess(analysisDir):
   #description: This function is the initial preprocessing of the subjects that will
    #be included in the data analysis
    #the subjects removed have technical problems in their initial values and analysis
    #return value: numpy array shape (number of subjects after preprocessing,1)

    subject_names = os.listdir(analysisDir)
    subject_names = [i for i in subject_names if (search('H\d\d_[A-Z][A-Z]',i) or search('H\d\d\d_[A-Z][A-Z]',i)) ]
    subject_names.sort()
    del subject_names[1:8]

    subject_names.remove('H010_AG')
    subject_names.remove('H011_GP')
    subject_names.remove('H014_ZW')
    subject_names.remove('H029_ON')
    subject_names.remove('H057_YP')
    subject_names.remove('H60_GG')
    subject_names.remove('H061_SE')
    subject_names=np.array(subject_names)
    print(subject_names)
    subject_names =subject_names.reshape(-1,1)
    return subject_names

subject_names = HUJI_subjects_preprocess(analysisDir)



# SECOND PART
def get_subregions_col(r_address, l_address):
    #this function creates a panda dataframe of one column with the names of all of the cortical
    #regions included in the analysis
    #input: file address of the a text file which includes the names of the cortical ROIs for
    #right hemisphere amd left hemisphere
    fid = open(l_address, 'r')
    lh_headers = fid.readline()
    fid.close()
    lh_headers = lh_headers.split()
    lh_headers = lh_headers[1:]


    fid = open(r_address, 'r')
    rh_headers = fid.readline()
    fid.close()
    rh_headers = rh_headers.split()
    rh_headers = rh_headers[1:]

    column_name = ["subregions"]

    names = lh_headers + rh_headers
    names = np.array(names)
    #names = np.zeros((72, 1))
    names =names.reshape(-1,1)
    names = pd.DataFrame(names, columns = column_name)
    print(names)
    dim = len(rh_headers) + len(lh_headers)
    return names

l_address= '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/statsTmp_lh.txt'
r_address = '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/statsTmp_rh.txt'
get_subregions_col(r_address, l_address)

# count_miss = 0
# count_not_miss = 0
# val_subjects = []
# # names = dict()
# for sub in range(len(subject_names)):
#     subject = subject_names[sub][0]
#
#     if (subject == 'H043_OS'):  # why do we not use this subject?
#         continue
#     # make sure we're not wasting time
#     os.chdir(analysisDir)
#     scanedate = os.path.abspath(subject)
#     os.chdir(scanedate)
#     readmepath = os.path.relpath('readme', scanedate)
#     if os.path.isfile(readmepath):  # has a read me
#         file1 = open(readmepath, 'r')
#         A = file1.readlines()[0]
#         # print(A)
#         sub_path = scanedate + '/' + A
#
#     else:
#         subfolders = glob.glob(scanedate + '/*/')
#         if subfolders == []:
#             continue
#         sub_path = subfolders[0]
#         # print(sub_path)
#     T1file = os.path.join(sub_path, 'mrQ_fixbias', 'OutPutFiles_1', 'BrainMaps', 'T1_map_Wlin.nii.gz')
#     TVfile = os.path.join(sub_path, 'mrQ_fixbias', 'OutPutFiles_1', 'BrainMaps', 'TV_map.nii.gz')
#     R2sfile = os.path.join(sub_path, 'multiecho_flash_R2s', 'R2_mean_2TVfixB1.nii.gz')
#     MTfile = os.path.join(sub_path, 'MT', 'MT_sat_mrQ_fixbias.nii.gz')
#     segfile = os.path.join(sub_path, 'freesurfer', 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz')
#
#     if not os.path.isfile(T1file) or not os.path.isfile(TVfile) or not os.path.isfile(R2sfile) or not os.path.isfile(
#             MTfile) or not os.path.isfile(segfile):
#         count_miss += 1
#         print(sub_path, "out")
#         continue
#     count_not_miss += 1
#
#     maps = [T1file, TVfile, R2sfile, MTfile]
#     count += 1
#
#     # collent the data
#     # cortical
#     cur_stats = []
#     for i in range(1, 3):
#
#         if i == 1:
#             hemi = 'lh'
#         if i == 2:
#             hemi = 'rh'
#
#             # once other types of calculations exist, put a flag here, this is the average for the 3rd layer of the cortex
#         outputFile = '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/statsTmp_' + hemi + '.txt'
#         statsDir = os.path.join(fsMainDir, subject)
#         measure = 'thickness'
#         cmd = 'aparcstats2table --subjects ' + statsDir + ' --hemi ' + hemi + ' --meas ' + measure + ' --parc ' + parcName + ' --tablefile ' + outputFile
#         print(cmd, "cmd")
#         status = os.system(cmd)
#         print(status)
#         # here call functions that will do other types of calculations using numpy or pandas
#
#         # all this can be extracted to a function to be more organized
#
#         # extract the stats values for each hemisphere
#         # file1 = open(readmepath, 'r')
#         # A = file1.readlines()[0]
#         fid = open(outputFile, 'r')
#         line1 = fid.readline()
#
#         values = fid.readline()
#
#         fid.close()
#
#         # if count!=33:
#         #    delete(outputFile)
#         values = values.split()
#         values = values[1:]
#
#         # cur_stats+=values
#         cur_stats.append(values)
#         # print(cur_stats, "curstats")
#         # now change to pandas
#         # df = pd.DataFrame(line2)
#
#     # nibabel:
#     # ni_image = nibabel.load(T1file)
#     # img_array = ni_image.get_fdata()
#
#     cur_stats = np.array(cur_stats)
#     cur_stats = cur_stats.reshape(-1, 1)
#     cur_stats = pd.DataFrame(cur_stats, columns=[subject_names[sub]])
#     names[subject_names[sub][0]] = cur_stats
#
#     # print(cur_stats,"name",subject_names[sub])
#     val_subjects.append(subject_names[sub])
#     # np.append(statsVec,[cur_stats], axis=1)
# # names = names.drop(names.columns[0], axis=1)
# df_sub_names = pd.DataFrame(val_subjects)
# df_sub_names.to_csv('/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/sub_names2.csv')
# names.to_csv('/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/midgray_MT_sec_version.csv')
# val_subjects.sort()
# # print(val_subjects)
# # print(len(val_subjects))
# # print(names)
#

# finding the file path for each subject
# this function also appears in Calc_Cov_Huji


analysisDir = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'


def get_subject_paths(analysisDir, niba_file_name, subject_names):
    # analysisDir: the directory of the dataset used
    # niba_file_name (nii.gz ending)
    # will create an array of the sub_path of each subject
    # ex1: analysisDir: '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
    # niba_file_name: 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz'
    # subject_names: numpy array of names of the subjects: shape:(num of subjects, 1), vector, return value from
    # HUJI_subjects_preprocess
    # ex2: analysisDir: '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
    # niba_file_name: 'first_all_none_firstseg.nii.gz'
    subject_paths = []
    for sub in range(len(subject_names)):
        subject = subject_names[sub]
        os.chdir(analysisDir)
        scanedate = os.path.join(analysisDir, subject[0])
        os.chdir(scanedate)
        readmepath = os.path.relpath('readme', scanedate)
        if os.path.isfile(readmepath):  # has a read me
            file1 = open(readmepath, 'r')
            A = file1.readlines()[0]
            sub_path = scanedate + '/' + A

        else:
            subfolders = glob.glob(scanedate + '/*/')
            if subfolders == []:
                continue

            sub_path = subfolders[0]
        subject_paths.append(sub_path)
    return subject_paths

# get_subject_paths(analysisDir,'first_all_none_firstseg.nii.gz', names_np) #works
# get_subject_paths(analysisDir, 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz', subject_names) #works


# this function also appears in Calc_Cov_Huji
def measures_per_subject(sub_path):
    # Creates np array of np arrays of the parameters data, t1,r2s,mt, tv and seg file
    # input: sub_path: string representing the folder path of a subject to their MRI scan data

    # look for maps
    T1file = os.path.join(sub_path, 'mrQ_fixbias', 'OutPutFiles_1', 'BrainMaps', 'T1_map_Wlin.nii.gz')
    TVfile = os.path.join(sub_path, 'mrQ_fixbias', 'OutPutFiles_1', 'BrainMaps', 'TV_map.nii.gz')
    R2sfile = os.path.join(sub_path, 'multiecho_flash_R2s', 'R2_mean_2TVfixB1.nii.gz')
    MTfile = os.path.join(sub_path, 'MT', 'MT_sat_mrQ_fixbias.nii.gz');
    segfile = os.path.join(sub_path, 'freesurfer', 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz')

    if not os.path.isfile(T1file) or not os.path.isfile(TVfile) or not os.path.isfile(R2sfile) or not os.path.isfile(
            MTfile) or not os.path.isfile(segfile):
        return -1, -1  # -1 acts as a return code
    # sub_names.append(names_np[name_idx][0])
    seg = nib.load(segfile)
    t1 = nib.load(T1file)
    # r1 = 1/t1.get_fdata() #problem with zero division, change later
    t1 = t1.get_fdata()

    tv = nib.load(TVfile)
    tv = tv.get_fdata()

    r2s = nib.load(R2sfile)
    r2s = r2s.get_fdata()

    mt = nib.load(MTfile)
    mt = mt.get_fdata()
    seg = seg.get_fdata()

    measures = [t1, tv, r2s, mt]  # change back to r1 once you know how to fix the division by zero
    return measures, seg


# refractored (to some degree)
subject_names = subject_names.reshape(-1, 1)


analysisDir = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/HUJI/Calibration/Human'
subject_paths = get_subject_paths(analysisDir, 'segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz', subject_names)
fsMainDir = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/freesurfer_subjects/'


def collect_midgray_stats(fsMainDir, subject_paths, subject_names, parcName):
    # decription: This function
    val_subjects = []
    sub_names = []
    name_idx = 0
    names={}
    for sub_path in subject_paths:
        measures, seg = measures_per_subject(sub_path)

        if measures == -1:
            name_idx += 1
            continue
        sub_names.append(subject_names[name_idx][0])
        name_idx += 1

        # collent the data
        # cortical
        cur_stats = []
        for i in range(1, 3):

            if i == 1:
                hemi = 'lh'
            if i == 2:
                hemi = 'rh'

                # once other types of calculations exist, put a flag here, this is the average for the 3rd layer of the cortex
            outputFile = '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/statsTmp_' + hemi + '.txt'
            # outputFile = '/tmp/statsTmp_' + hemi + '.txt'
            # statsDir = os.path.join(fsMainDir, subject)
            statsDir = os.path.join(fsMainDir, subject_names[name_idx-1][0])
            measure = 'thickness'
            cmd = 'aparcstats2table --subjects ' + statsDir + ' --hemi ' + hemi + ' --meas ' + measure + ' --parc ' + parcName + ' --tablefile ' + outputFile

            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            status = os.system(cmd)
            print(cmd)
            # extract the stats values for each hemisphere
            fid = open(outputFile, 'r')
            line1 = fid.readline()
            values = fid.readline()
            fid.close()

            values = values.split()
            values = values[1:]
            cur_stats += values

        cur_stats = np.array(cur_stats)
        cur_stats = cur_stats.reshape(-1, 1)
        cur_stats = pd.DataFrame(cur_stats, columns=[subject_names[name_idx-1][0]])
        names[subject_names[name_idx-1][0]] = cur_stats
        val_subjects.append(subject_names[name_idx-1])

    df_sub_names = pd.DataFrame(val_subjects)
    df_sub_names.to_csv('/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/sub_names_try.csv')
    names.to_csv('/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/' + parcName + '_try.csv')
    val_subjects.sort()


# recall this function for each parameter instead of if/else
collect_midgray_stats(fsMainDir, subject_paths, subject_names, parcName)