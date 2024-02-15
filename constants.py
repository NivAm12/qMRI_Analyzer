import enum
import os
from dotenv import load_dotenv
load_dotenv()


# ------------------------- PATHS ------------------------- #
# ---------------- Paths to raw data ----------------- #
PATH_TO_RAW_DATA = os.getenv("PATH_TO_RAW_DATA")
PATH_TO_RAW_DATA_6_PARAMS = os.getenv("PATH_TO_RAW_DATA_6_PARAMS")
PATH_TO_RAW_DATA_ROBUST_SCALED = os.getenv("PATH_TO_RAW_DATA_ROBUST_SCALED")
PATH_TO_RAW_DATA_Z_SCORED = os.getenv("PATH_TO_RAW_DATA_Z_SCORED")
PATH_TO_CORTEX_4_PARAMS_RAW = os.getenv("PATH_TO_CORTEX_4_PARAMS_RAW")
PATH_TO_CORTEX_4_PARAMS_Z = os.getenv("PATH_TO_CORTEX_4_PARAMS_Z")
PATH_TO_CORTEX_Z_SCORED = os.getenv("PATH_TO_CORTEX_Z_SCORED")
PATH_TO_FRONTAL_CORTEX_6_params = os.getenv("PATH_TO_FRONTAL_CORTEX_6_params")
PATH_TO_FRONTAL_CORTEX_4_params = os.getenv("PATH_TO_FRONTAL_CORTEX_4_params")
PATH_TO_RIGHT_CORTEX_4_params_ZSCORE = os.getenv("PATH_TO_RIGHT_CORTEX_4_params_ZSCORE")
PATH_TO_FRONTAL_CORTEX_4_params_RAW = os.getenv("PATH_TO_FRONTAL_CORTEX_4_params_RAW")
PATH_TO_PUTAMEN_CAUDETE_6_params = os.getenv("PATH_TO_PUTAMEN_CAUDETE_6_params")
PATH_TO_PUTAMEN_THALAMUS_6_params = os.getenv("PATH_TO_PUTAMEN_THALAMUS_6_params")
PATH_TO_PALLIDUM_PUTAMEN_CAUDETE_6_params = os.getenv("PATH_TO_PALLIDUM_PUTAMEN_CAUDETE_6_params")
PATH_TO_AMYGDALA_HIPPOCAMPUS_6_params = os.getenv("PATH_TO_AMYGDALA_HIPPOCAMPUS_6_params")
PATH_TO_ACCUM_HIPPO_AMYG_6_params = os.getenv("PATH_TO_ACCUM_HIPPO_AMYG_6_params")
PATH_TO_CORTEX_all_params_z_score = os.getenv('PATH_TO_CORTEX_all_params_z_score')
PATH_TO_CORTEX_all_params_raw = os.getenv('PATH_TO_CORTEX_all_params_raw')
SUBJECTS_INFO_PATH = os.getenv("SUBJECTS_INFO_PATH")

# ---------------- Paths to output data ----------------- #
SAVE_DATA_PATH = os.getenv("SAVE_DATA_PATH")
SAVE_DATA_OF_ADULTS_Z_SCORE_MEANS = os.getenv("SAVE_DATA_OF_ADULTS_Z_SCORE_MEANS")
SAVE_DATA_OF_YOUNG_Z_SCORE_MEANS = os.getenv("SAVE_DATA_OF_YOUNG_Z_SCORE_MEANS")
SAVE_DATA_OF_ALL_Z_SCORE_MEANS = os.getenv("SAVE_DATA_OF_ALL_Z_SCORE_MEANS")

# ---------------- Paths to subject data ----------------- #
ANALYSIS_DIR = os.getenv('ANALYSIS_DIR')
OLDER_ANALYSIS_DIR = os.getenv('OLDER_ANALYSIS_DIR')
ANALYSIS_DIR_Par = os.getenv('ANALYSIS_DIR_Par')
CLUSTERING_PATH = os.getenv('CLUSTERING_PATH')

# -------------------- File Names -------------------- #
HIERARCHICAL_CLUSTERING_FILE = "corr_hirr_clustering_"

# -------------------- Statistics Funcs to Run -------------------- #
T_TEST = 't_tests'
HIERARCHICAL_CLUSTERING_WITH_CORRELATIONS = 'hierarchical_clustering_with_correlations'
SD_PER_PARAMETER = 'sd_per_parameter'
PLOT_DATA_PER_PARAM = 'plot data per param'
PLOT_DATA_PER_ROI_PER_SUBJET_WITH_ALL_PARAMS = 'PLOT_DATA_PER_ROI_PER_SUBJET_WITH_ALL_PARAMS'
HIERARCHICAL_CLUSTERING = 'hierarchical_clustering'
ROIS_CORRELATIONS = 'rois_correlations'
PLOT_BRAIN_CLUSTERS = 'plot_brain_clusters'
LINKAGE_METRICS = ['complete', 'average', 'ward', 'median']

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

ROI_CORTEX = {
    1002: 'ctx-lh-caudalanteriorcingulate',
    1003: 'ctx-lh-caudalmiddlefrontal',
    1005: 'ctx-lh-cuneus',
    1006: 'ctx-lh-entorhinal',
    1007: 'ctx-lh-fusiform',
    1008: 'ctx-lh-inferiorparietal',
    1009: 'ctx-lh-inferiortemporal',
    1010: 'ctx-lh-isthmuscingulate',
    1011: 'ctx-lh-lateraloccipital',
    1012: 'ctx-lh-lateralorbitofrontal',
    1013: 'ctx-lh-lingual',
    1014: 'ctx-lh-medialorbitofrontal',
    1015: 'ctx-lh-middletemporal',
    1016: 'ctx-lh-parahippocampal',
    1017: 'ctx-lh-paracentral',
    1018: 'ctx-lh-parsopercularis',
    1019: 'ctx-lh-parsorbitalis',
    1020: 'ctx-lh-parstriangularis',
    1021: 'ctx-lh-pericalcarine',
    1022: 'ctx-lh-postcentral',
    1023: 'ctx-lh-posteriorcingulate',
    1024: 'ctx-lh-precentral',
    1025: 'ctx-lh-precuneus',
    1026: 'ctx-lh-rostralanteriorcingulate',
    1027: 'ctx-lh-rostralmiddlefrontal',
    1028: 'ctx-lh-superiorfrontal',
    1029: 'ctx-lh-superiorparietal',
    1030: 'ctx-lh-superiortemporal',
    1031: 'ctx-lh-supramarginal',
    1034: 'ctx-lh-transversetemporal',
    1035: 'ctx-lh-insula',
    2002: 'ctx-rh-caudalanteriorcingulate',
    2003: 'ctx-rh-caudalmiddlefrontal',
    2005: 'ctx-rh-cuneus',
    2007: 'ctx-rh-fusiform',
    2008: 'ctx-rh-inferiorparietal',
    2009: 'ctx-rh-inferiortemporal',
    2010: 'ctx-rh-isthmuscingulate',
    2011: 'ctx-rh-lateraloccipital',
    2012: 'ctx-rh-lateralorbitofrontal',
    2013: 'ctx-rh-lingual',
    2014: 'ctx-rh-medialorbitofrontal',
    2015: 'ctx-rh-middletemporal',
    2016: 'ctx-rh-parahippocampal',
    2017: 'ctx-rh-paracentral',
    2018: 'ctx-rh-parsopercularis',
    2019: 'ctx-rh-parsorbitalis',
    2020: 'ctx-rh-parstriangularis',
    2021: 'ctx-rh-pericalcarine',
    2022: 'ctx-rh-postcentral',
    2023: 'ctx-rh-posteriorcingulate',
    2024: 'ctx-rh-precentral',
    2025: 'ctx-rh-precuneus',
    2026: 'ctx-rh-rostralanteriorcingulate',
    2027: 'ctx-rh-rostralmiddlefrontal',
    2028: 'ctx-rh-superiorfrontal',
    2029: 'ctx-rh-superiorparietal',
    2030: 'ctx-rh-superiortemporal',
    2031: 'ctx-rh-supramarginal',
    2034: 'ctx-rh-transversetemporal',
    2035: 'ctx-rh-insula'
}


ROI_FRONTAL_CORTEX = {
    1003: 'ctx-lh-caudalmiddlefrontal',
    2003: 'ctx-rh-caudalmiddlefrontal',
    1012: 'ctx-lh-lateralorbitofrontal',
    2012: 'ctx-rh-lateralorbitofrontal',
    1017: 'ctx-lh-paracentral',
    2017: 'ctx-rh-paracentral',
    1018: 'ctx-lh-parsopercularis',
    2018: 'ctx-rh-parsopercularis',
    1019: 'ctx-lh-parsorbitalis',
    2019: 'ctx-rh-parsorbitalis',
    1020: 'ctx-lh-parstriangularis',
    2020: 'ctx-rh-parstriangularis',
    1024: 'ctx-lh-precentral',
    2024: 'ctx-rh-precentral',
    1027: 'ctx-lh-rostralmiddlefrontal',
    2027: 'ctx-rh-rostralmiddlefrontal',
    1028: 'ctx-lh-superiorfrontal',
    2028: 'ctx-rh-superiorfrontal',
    1032: 'ctx-lh-frontalpole',
    2032: 'ctx-rh-frontalpole',
}

ROI_LEFT_CORTEX = {
    1000: 'ctx-lh-unknown',
    1001: 'ctx-lh-bankssts',
    1002: 'ctx-lh-caudalanteriorcingulate',
    1003: 'ctx-lh-caudalmiddlefrontal',
    1004: 'ctx-lh-corpuscallosum',
    1005: 'ctx-lh-cuneus',
    1006: 'ctx-lh-entorhinal',
    1007: 'ctx-lh-fusiform',
    1008: 'ctx-lh-inferiorparietal',
    1009: 'ctx-lh-inferiortemporal',
    1010: 'ctx-lh-isthmuscingulate',
    1011: 'ctx-lh-lateraloccipital',
    1012: 'ctx-lh-lateralorbitofrontal',
    1013: 'ctx-lh-lingual',
    1014: 'ctx-lh-medialorbitofrontal',
    1015: 'ctx-lh-middletemporal',
    1016: 'ctx-lh-parahippocampal',
    1017: 'ctx-lh-paracentral',
    1018: 'ctx-lh-parsopercularis',
    1019: 'ctx-lh-parsorbitalis',
    1020: 'ctx-lh-parstriangularis',
    1021: 'ctx-lh-pericalcarine',
    1022: 'ctx-lh-postcentral',
    1023: 'ctx-lh-posteriorcingulate',
    1024: 'ctx-lh-precentral',
    1025: 'ctx-lh-precuneus',
    1026: 'ctx-lh-rostralanteriorcingulate',
    1027: 'ctx-lh-rostralmiddlefrontal',
    1028: 'ctx-lh-superiorfrontal',
    1029: 'ctx-lh-superiorparietal',
    1030: 'ctx-lh-superiortemporal',
    1031: 'ctx-lh-supramarginal',
    1032: 'ctx-lh-frontalpole',
    1033: 'ctx-lh-temporalpole',
    1034: 'ctx-lh-transversetemporal',
    1035: 'ctx-lh-insula'
}


ROI_RIGTH_CORTEX = {
    2000: 'ctx-rh-unknown',
    2001: 'ctx-rh-bankssts',
    2002: 'ctx-rh-caudalanteriorcingulate',
    2003: 'ctx-rh-caudalmiddlefrontal',
    2004: 'ctx-rh-corpuscallosum',
    2005: 'ctx-rh-cuneus',
    2006: 'ctx-rh-entorhinal',
    2007: 'ctx-rh-fusiform',
    2008: 'ctx-rh-inferiorparietal',
    2009: 'ctx-rh-inferiortemporal',
    2010: 'ctx-rh-isthmuscingulate',
    2011: 'ctx-rh-lateraloccipital',
    2012: 'ctx-rh-lateralorbitofrontal',
    2013: 'ctx-rh-lingual',
    2014: 'ctx-rh-medialorbitofrontal',
    2015: 'ctx-rh-middletemporal',
    2016: 'ctx-rh-parahippocampal',
    2017: 'ctx-rh-paracentral',
    2018: 'ctx-rh-parsopercularis',
    2019: 'ctx-rh-parsorbitalis',
    2020: 'ctx-rh-parstriangularis',
    2021: 'ctx-rh-pericalcarine',
    2022: 'ctx-rh-postcentral',
    2023: 'ctx-rh-posteriorcingulate',
    2024: 'ctx-rh-precentral',
    2025: 'ctx-rh-precuneus',
    2026: 'ctx-rh-rostralanteriorcingulate',
    2027: 'ctx-rh-rostralmiddlefrontal',
    2028: 'ctx-rh-superiorfrontal',
    2029: 'ctx-rh-superiorparietal',
    2030: 'ctx-rh-superiortemporal',
    2031: 'ctx-rh-supramarginal',
    2032: 'ctx-rh-frontalpole',
    2033: 'ctx-rh-temporalpole',
    2034: 'ctx-rh-transversetemporal',
    2035: 'ctx-rh-insula'
}

DICT_NUM_TO_ROIS = {1: SUB_CORTEX_DICT,
                    2: ROI_PUTAMEN_THALAMUS,
                    3: ROI_PALLIDUM_PUTAMEN_CAUDETE,
                    4: ROI_PUTAMEN_CAUDETE,
                    5: ROI_AMYGDALA_HIPPOCAMPUS,
                    6: ROI_ACCUM_HIPPO_AMYG}

# -------------------- MRI Physical Parameters -------------------- #
BASIC_4_PARAMS = ["r1", "tv", "r2s", "mt"]
BASIC_4_PARAMS_WITH_SLOPES = ["r1", "tv", "r2s", "mt", "Slope-tv-r1", "Slope-tv-r2s", "Dtv-r1-values", "Dtv-r2s-values"]
ALL_PARAMS_WITH_SLOPES = ["r1", "tv", "r2s", "mt", "t2", "diffusion_fa", "diffusion_md",
                          "Slope-tv-r1", "Slope-tv-r2s", "Slope-tv-mt", "Slope-tv-t2",
                          "Slope-tv-diffusion_fa", "Slope-tv-diffusion_md", "Slope-r2s-r1"]

PARAMETERS = ["r1", "tv", "r2s", "mt", "t2", "diffusion"]
PARAMETERS_W_D_TV_R1_AND_R2S = ["r1", "tv", "r2s", "mt", "t2", "diffusion", "Slope-tv-r1", "Slope-tv-r2s",
                                "Dtv-r1-values", "Dtv-r2s-values"]
PARAMS_OF_SLOPES = ["Slope-tv-r1", "Slope-tv-r2s", "Dtv-r1-values", "Dtv-r2s-values"]

# -------------------- Groups Dividers Consts -------------------- #
OLD = "OLD"
YOUNG = "YOUNG"
AGE_THRESHOLD = 55

# -------------------- Enums for statistical actions -------------------- #
class Actions(enum.Enum):
    z_score = 1  # Z Score on data - per subject, per parameter, per ROI
    z_score_means = 2  # Z Score on means of all subjects, per parameters, per its ROI
    means_per_subject = 3  # Means on each Subject, per its ROI (per parameter)
    robust_scaling = 4  # subtracting the median and dividing by the interquartile range


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
DIFFUSION_MD = 'diffusion_md'
DIFFUSION_FA = 'diffusion_fa'
T2 = 't2'

# qMRI parameter's maps
MAP_T2 = os.getenv("MAP_T2")
MAP_T2_TRANSFORMED = os.getenv('MAP_T2_TRANSFORMED')
MAP_DIFFUSION = os.getenv("MAP_DIFFUSION")
MAP_DIFFUSION_FA = os.getenv('MAP_DIFFUSION_FA')
MAP_DIFFUSION_MD = os.getenv('MAP_DIFFUSION_MD')
MAP_DIFFUSION_MD_TRANSFORMED = os.getenv('MAP_DIFFUSION_MD_TRANSFORMED')
MAP_DIFFUSION_FA_TRANSFORMED = os.getenv('MAP_DIFFUSION_FA_TRANSFORMED')
MAP_R1 = os.getenv("MAP_R1")
MAP_TV = os.getenv("MAP_TV")
MAP_MT = os.getenv("MAP_MT")
MAP_R2S = os.getenv("MAP_R2S")

# qMRI segmentation's maps
BASIC_SEG = os.getenv("BASIC_SEG")
SEG_T2 = os.getenv("SEG_T2")
SEG_DIFFUSION = os.getenv("SEG_DIFFUSION")

# Plots values
COLOR_LIST = {
    0: 3,
    1: 1016,
    2: 1024,
    3: 26
}

WANDB_ENTITY = os.getenv('WANDB_ENTITY')