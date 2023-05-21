import enum

# ------------------------- PATHS ------------------------- #
# ---------------- Paths to raw data ----------------- #
PATH_TO_RAW_DATA = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/corr_by_means" \
                   "/subcortical_updated/with_R1/raw_data_of_subjects/raw_data"
PATH_TO_RAW_DATA_6_PARAMS = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/corr_by_means" \
                   "/subcortical_updated/with_R1/raw_data_of_subjects/raw_data_6_params"

PATH_TO_RAW_DATA_ROBUST_SCALED = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/corr_by_means" \
                                "/subcortical_updated/with_R1/raw_data_of_subjects/raw_data_robust_scaling3"

PATH_TO_RAW_DATA_Z_SCORED = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/corr_by_means" \
                            "/subcortical_updated/with_R1/raw_data_of_subjects/raw_data_z_score_on_brain3"

SUBJECTS_INFO_PATH = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/subjects_info.csv"

# ---------------- Paths to output data ----------------- #
SAVE_DATA_PATH = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/saved_versions/corr_by_means/" \
                 "subcortical_updated/with_R1/Analysis/young_adults_comparison/"

SAVE_DATA_OF_ADULTS_Z_SCORE_MEANS = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/" \
                                    "Covariance_Aging/saved_versions/corr_by_means/" \
                                    "subcortical_updated/with_R1/Analysis/young_adults_comparison/" \
                                    "default_z_score_on_average_of_params/calculation/adults/"

SAVE_DATA_OF_YOUNG_Z_SCORE_MEANS = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/" \
                                    "Covariance_Aging/saved_versions/corr_by_means/" \
                                    "subcortical_updated/with_R1/Analysis/young_adults_comparison/" \
                                    "default_z_score_on_average_of_params/calculation/young/"

SAVE_DATA_OF_ALL_Z_SCORE_MEANS = "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/" \
                                    "Covariance_Aging/saved_versions/corr_by_means/" \
                                    "subcortical_updated/with_R1/Analysis/young_adults_comparison/" \
                                    "default_z_score_on_average_of_params/calculation/all/"


# -------------------- File Names -------------------- #
HIERARCHICAL_CLUSTERING_FILE = "corr_hirr_clustering_"


# -------------------- Statistics Funcs to Run -------------------- #
T_TEST = 't_tests'
HIERARCHICAL_CLUSTERING_WITH_CORRELATIONS = 'hierarchical_clustering_with_correlations'
SD_PER_PARAMETER = 'sd_per_parameter'
PLOT_DATA_PER_PARAM = 'plot data per param'
PLOT_DATA_PER_ROI_PER_SUBJET_WITH_ALL_PARAMS = 'PLOT_DATA_PER_ROI_PER_SUBJET_WITH_ALL_PARAMS'


# -------------------- Folders Output ---------------------- #
Z_SCORE_ON_AVG_ON_BRAIN_DIR = "default_z_score_on_average_of_params/"
Z_SCORE_ON_BRAIN_DIR = "z_score_on_brain/"
NORMALIZE_BY_MEDIAN_DIR = "normalize_by_median_on_brain/"
MEANS_ON_BRAIN_DIR = "average_of_params/"
STD_OF_PARAMS_BRAIN_DIR = "std_of_params/"

# -------------- Sub Folders - by ROIS ------------- #
PALLIDUM_PUTAMEN_CAUDETE_DIR = "pallidum_putamen_caudete/"
PUTAMEN_CAUDETE_DIR = "putamen_caudete/"
AMYGDALA_HIPPOCAMPUS_DIR = "amygdala_hippocampus/"

# -------------- Sub Folders - by Raw Data Type ------------- #
RAW_DATA_DIR = "raw_data/"
RAW_DATA_6_PARAMS_DIR = "raw_data_6_params/"
RAW_DATA_Z_SCORED_DIR = "raw_data_z_scored/"
RAW_DATA_ROBUST_SCALED_DIR = "raw_data_robust_scaled/"

# -------------------- ROIs -------------------- #
SUB_CORTEX_DICT = {10: 'Left-Thalamus-Proper', 11: 'Left-Caudate', 12: 'Left-Putamen', 13: 'Left-Pallidum',
                   17: 'Left-Hippocampus', 18: 'Left-Amygdala', 26: 'Left-Accumbens-area',
                   49: 'Right-Thalamus-Proper', 50: 'Right-Caudate', 51: 'Right-Putamen', 52: 'Right-Pallidum',
                   53: 'Right-Hippocampus', 54: 'Right-Amygdala',
                   58: 'Right-Accumbens-area'}  # TODO: GOOD VERSION DO NOT DELETE!!!!

ROI_PUTAMEN_THALAMUS = {10: 'Left-Thalamus-Proper', 12: 'Left-Putamen',
                        49: 'Right-Thalamus-Proper', 51: 'Right-Putamen'}

ROI_PALLIDUM_PUTAMEN_CAUDETE = {11: 'Left-Caudate', 12: 'Left-Putamen', 13: 'Left-Pallidum', 50: 'Right-Caudate',
                                51: 'Right-Putamen', 52: 'Right-Pallidum'}

ROI_AMYGDALA_HIPPOCAMPUS = {17: 'Left-Hippocampus', 18: 'Left-Amygdala', 53: 'Right-Hippocampus',
                            54: 'Right-Amygdala'}

ROI_PUTAMEN_CAUDETE = {11: 'Left-Caudate', 12: 'Left-Putamen', 50: 'Right-Caudate', 51: 'Right-Putamen'}

ROI_ACCUM_HIPPO_AMYG = {17: 'Left-Hippocampus', 18: 'Left-Amygdala', 26: 'Left-Accumbens-area',
                        53: 'Right-Hippocampus', 54: 'Right-Amygdala', 58: 'Right-Accumbens-area'}

DICT_NUM_TO_ROIS = {1: SUB_CORTEX_DICT,
                    2: ROI_PUTAMEN_THALAMUS,
                    3: ROI_PALLIDUM_PUTAMEN_CAUDETE,
                    4: ROI_PUTAMEN_CAUDETE,
                    5: ROI_AMYGDALA_HIPPOCAMPUS,
                    6: ROI_ACCUM_HIPPO_AMYG}

# -------------------- MRI Physical Parameters -------------------- #
BASIC_4_PARAMS = ["r1", "tv", "r2s", "mt"]
PARAMETERS = ["r1", "tv", "r2s", "mt", "t2", "diffusion"]
PARAMETERS_W_D_TV_R1_AND_R2S = ["r1", "tv", "r2s", "mt", "t2", "diffusion", "Slope-tv-r1", "Slope-tv-r2s", "Dtv-r1-values", "Dtv-r2s-values"]
PARAMS_OF_SLOPES = ["Slope-tv-r1", "Slope-tv-r2s", "Dtv-r1-values", "Dtv-r2s-values"]

# -------------------- Groups Dividers Consts -------------------- #
OLD = "OLD"
YOUNG = "YOUNG"
AGE_THRESHOLD = 40


# -------------------- Enums for statistical actions -------------------- #
class Actions(enum.Enum):
    z_score = 1             # Z Score on data - per subject, per parameter, per ROI
    z_score_means = 2       # Z Score on means of all subjects, per parameters, per its ROI
    means_per_subject = 3   # Means on each Subject, per its ROI (per parameter)
    robust_scaling = 4      # subtracting the median and dividing by the interquartile range


# -------------------- Statistical Methods Names -------------------- #
ROBUST_SCALING = "ROBUST_SCALING"
Z_SCORE = "Z_SCORE"
RAW_DATA = "RAW_DATA"
RAW_DATA_6_PARAMS = "RAW_DATA_6_PARAMS"

DICT_NUM_TO_METHOD = {1: RAW_DATA,
                      2: Z_SCORE,
                      3: ROBUST_SCALING}

# -------------------- MAPS and Segmentations paths -------------------- #
# qMRI parameter
R1 = 'r1'
R2S = 'r2s'
MT = 'mt'
TV = 'tv'
DIFFUSION = 'diffusion'
T2 = 't2'

# qMRI parameter's maps
MAP_R1 = 'mrQ_fixbias/OutPutFiles_1/BrainMaps/T1_map_Wlin.nii.gz'
MAP_R2S = 'multiecho_flash_R2s/R2_mean_2TVfixB1.nii.gz'
MAP_MT = 'MT/MT_sat_mrQ_fixbias.nii.gz'
MAP_TV = 'mrQ/OutPutFiles_1/BrainMaps/TV_correctedForT2s.nii.gz'
MAP_T2 = 'T2/T2map.nii.gz'
MAP_DIFFUSION = 'Dif_fsl_preprocessed/eddy/aligned2T1/dtiInit/dti94trilin/bin/MD.nii.gz'

# qMRI segmentation's maps
BASIC_SEG = 'freesurfer/segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL.nii.gz'
SEG_T2 = 'T2/segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL_2T2_resamp_BMnewmrQ.nii.gz'
SEG_DIFFUSION = 'Dif_fsl_preprocessed/eddy/aligned2T1/dtiInit/dti94trilin/bin/segFSLMPRAGE_BS_wmparc2newmrQ_B1corrALL_2DTI_resamp.nii.gz'
