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
PATH_TO_RIGHT_CORTEX_4_params_ZSCORE = os.getenv(
    "PATH_TO_RIGHT_CORTEX_4_params_ZSCORE")
PATH_TO_FRONTAL_CORTEX_4_params_RAW = os.getenv(
    "PATH_TO_FRONTAL_CORTEX_4_params_RAW")
PATH_TO_PUTAMEN_CAUDETE_6_params = os.getenv(
    "PATH_TO_PUTAMEN_CAUDETE_6_params")
PATH_TO_PUTAMEN_THALAMUS_6_params = os.getenv(
    "PATH_TO_PUTAMEN_THALAMUS_6_params")
PATH_TO_PALLIDUM_PUTAMEN_CAUDETE_6_params = os.getenv(
    "PATH_TO_PALLIDUM_PUTAMEN_CAUDETE_6_params")
PATH_TO_AMYGDALA_HIPPOCAMPUS_6_params = os.getenv(
    "PATH_TO_AMYGDALA_HIPPOCAMPUS_6_params")
PATH_TO_ACCUM_HIPPO_AMYG_6_params = os.getenv(
    "PATH_TO_ACCUM_HIPPO_AMYG_6_params")
PATH_TO_CORTEX_all_params_z_score = os.getenv(
    'PATH_TO_CORTEX_all_params_z_score')
PATH_TO_CORTEX_all_params_raw = os.getenv('PATH_TO_CORTEX_all_params_raw')
PATH_TO_CORTEX_all_params_robust = os.getenv(
    'PATH_TO_CORTEX_all_params_robust')
PATH_TO_SUB_CORTEX_all_params_z_score = os.getenv(
    'PATH_TO_SUB_CORTEX_all_params_z_score')
PATH_TO_SUB_CORTEX_all_params_raw = os.getenv(
    'PATH_TO_SUB_CORTEX_all_params_raw')
PATH_TO_PD_CORTEX_all_params_raw = os.getenv(
    'PATH_TO_PD_CORTEX_all_params_raw')
PATH_TO_PD_CORTEX_all_params_z_score = os.getenv(
    'PATH_TO_PD_CORTEX_all_params_z_score')
PATH_TO_PD_SUB_CORTEX_all_params_raw = os.getenv(
    'PATH_TO_PD_SUB_CORTEX_all_params_raw')
PATH_TO_PD_SUB_CORTEX_all_params_z_score = os.getenv(
    'PATH_TO_PD_SUB_CORTEX_all_params_z_score')
PATH_TO_CORTEX_AND_GRAY_SUB_CORTEX_all_params_raw = os.getenv(
    'PATH_TO_CORTEX_AND_GRAY_SUB_CORTEX_all_params_raw')
PATH_TO_CORTEX_AND_GRAY_SUB_CORTEX_all_params_z_score = os.getenv(
    'PATH_TO_CORTEX_AND_GRAY_SUB_CORTEX_all_params_z_score')
SUBJECTS_INFO_PATH = os.getenv("SUBJECTS_INFO_PATH")
PATH_TO_WM_SUBCORTEX_all_params_raw = os.getenv(
    'PATH_TO_WM_SUBCORTEX_all_params_z_raw')
PATH_TO_WM_SUBCORTEX_all_params_z_score = os.getenv(
    'PATH_TO_WM_SUBCORTEX_all_params_z_score')
PATH_TO_ALL_BRAIN_all_params_raw = os.getenv(
    'PATH_TO_ALL_BRAIN_all_params_raw')
PATH_TO_ALL_BRAIN_all_params_z_score = os.getenv(
    'PATH_TO_ALL_BRAIN_all_params_z_score')

# ---------------- Paths to output data ----------------- #
SAVE_DATA_PATH = os.getenv("SAVE_DATA_PATH")
SAVE_DATA_OF_ADULTS_Z_SCORE_MEANS = os.getenv(
    "SAVE_DATA_OF_ADULTS_Z_SCORE_MEANS")
SAVE_DATA_OF_YOUNG_Z_SCORE_MEANS = os.getenv(
    "SAVE_DATA_OF_YOUNG_Z_SCORE_MEANS")
SAVE_DATA_OF_ALL_Z_SCORE_MEANS = os.getenv("SAVE_DATA_OF_ALL_Z_SCORE_MEANS")
SAVE_ADDRESS = os.getenv('SAVE_ADDRESS')

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

ROI_PUTAMEN_CAUDETE = {11: 'Left-Caudate', 12: 'Left-Putamen',
                       50: 'Right-Caudate', 51: 'Right-Putamen'}

ROI_ACCUM_HIPPO_AMYG = {17: 'Left-Hippocampus', 18: 'Left-Amygdala', 26: 'Left-Accumbens-area',
                        53: 'Right-Hippocampus', 54: 'Right-Amygdala', 58: 'Right-Accumbens-area'}
ROI_ALL_BRAIN = {
    1002: 'ctx-lh-caudalanteriorcingulate',
    1003: 'ctx-lh-caudalmiddlefrontal',
    1005: 'ctx-lh-cuneus',
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
    1018: 'ctx-lh-parsopercularis',
    1019: 'ctx-lh-parsorbitalis',
    1020: 'ctx-lh-parstriangularis',
    1022: 'ctx-lh-postcentral',
    1023: 'ctx-lh-posteriorcingulate',
    1024: 'ctx-lh-precentral',
    1025: 'ctx-lh-precuneus',
    1026: 'ctx-lh-rostralanteriorcingulate',
    1027: 'ctx-lh-rostralmiddlefrontal',
    1028: 'ctx-lh-superiorfrontal',
    1030: 'ctx-lh-superiortemporal',
    1031: 'ctx-lh-supramarginal',
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
    2018: 'ctx-rh-parsopercularis',
    2019: 'ctx-rh-parsorbitalis',
    2020: 'ctx-rh-parstriangularis',
    2022: 'ctx-rh-postcentral',
    2023: 'ctx-rh-posteriorcingulate',
    2024: 'ctx-rh-precentral',
    2025: 'ctx-rh-precuneus',
    2026: 'ctx-rh-rostralanteriorcingulate',
    2027: 'ctx-rh-rostralmiddlefrontal',
    2028: 'ctx-rh-superiorfrontal',
    2030: 'ctx-rh-superiortemporal',
    2031: 'ctx-rh-supramarginal',
    2035: 'ctx-rh-insula',
    10: 'Left-Thalamus',
    11: 'Left-Caudate',
    12: 'Left-Putamen',
    13: 'Left-Pallidum',
    17: 'Left-Hippocampus',
    18: 'Left-Amygdala',
    26: 'Left-Accumbens-area',
    28: 'Left-VentralDC',
    49: 'Right-Thalamus',
    50: 'Right-Caudate',
    51: 'Right-Putamen',
    52: 'Right-Pallidum',
    53: 'Right-Hippocampus',
    54: 'Right-Amygdala',
    58: 'Right-Accumbens-area',
    60: 'Right-VentralDC',
    3001: "wm-lh-bankssts",
    3002: "wm-lh-caudalanteriorcingulate",
    3003: "wm-lh-caudalmiddlefrontal",
    3005: "wm-lh-cuneus",
    3006: "wm-lh-entorhinal",
    3007: "wm-lh-fusiform",
    3008: "wm-lh-inferiorparietal",
    3009: "wm-lh-inferiortemporal",
    3010: "wm-lh-isthmuscingulate",
    3011: "wm-lh-lateraloccipital",
    3012: "wm-lh-lateralorbitofrontal",
    3013: "wm-lh-lingual",
    3014: "wm-lh-medialorbitofrontal",
    3015: "wm-lh-middletemporal",
    3016: "wm-lh-parahippocampal",
    3018: "wm-lh-parsopercularis",
    3019: "wm-lh-parsorbitalis",
    3020: "wm-lh-parstriangularis",
    3022: "wm-lh-postcentral",
    3023: "wm-lh-posteriorcingulate",
    3024: "wm-lh-precentral",
    3025: "wm-lh-precuneus",
    3026: "wm-lh-rostralanteriorcingulat",
    3027: "wm-lh-rostralmiddlefrontal",
    3028: "wm-lh-superiorfrontal",
    3030: "wm-lh-superiortemporal",
    3031: "wm-lh-supramarginal",
    3032: "wm-lh-frontalpole",
    3033: "wm-lh-temporalpole",
    3035: "wm-lh-insula",
    4001: "wm-rh-bankssts",
    4002: "wm-rh-caudalanteriorcingulate",
    4003: "wm-rh-caudalmiddlefrontal",
    4005: "wm-rh-cuneus",
    4006: "wm-rh-entorhinal",
    4007: "wm-rh-fusiform",
    4008: "wm-rh-inferiorparietal",
    4009: "wm-rh-inferiortemporal",
    4010: "wm-rh-isthmuscingulate",
    4011: "wm-rh-lateraloccipital",
    4012: "wm-rh-lateralorbitofrontal",
    4013: "wm-rh-lingual ",
    4014: "wm-rh-medialorbitofrontal",
    4015: "wm-rh-middletemporal",
    4016: "wm-rh-parahippocampal",
    4018: "wm-rh-parsopercularis",
    4019: "wm-rh-parsorbitalis",
    4020: "wm-rh-parstriangularis",
    4022: "wm-rh-postcentral",
    4023: "wm-rh-posteriorcingulate",
    4024: "wm-rh-precentral",
    4025: "wm-rh-precuneus",
    4026: "wm-rh-rostralanteriorcingulat",
    4027: "wm-rh-rostralmiddlefrontal",
    4028: "wm-rh-superiorfrontal",
    4030: "wm-rh-superiortemporal",
    4031: "wm-rh-supramarginal",
    4032: "wm-rh-frontalpole",
    4033: "wm-rh-temporalpole",
    4035: "wm-rh-insula"
}

ROI_CORTEX = {
    1002: 'ctx-lh-caudalanteriorcingulate',
    1003: 'ctx-lh-caudalmiddlefrontal',
    1005: 'ctx-lh-cuneus',
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
    # 1017: 'ctx-lh-paracentral',
    1018: 'ctx-lh-parsopercularis',
    1019: 'ctx-lh-parsorbitalis',
    1020: 'ctx-lh-parstriangularis',
    # 1021: 'ctx-lh-pericalcarine',
    1022: 'ctx-lh-postcentral',
    1023: 'ctx-lh-posteriorcingulate',
    1024: 'ctx-lh-precentral',
    1025: 'ctx-lh-precuneus',
    1026: 'ctx-lh-rostralanteriorcingulate',
    1027: 'ctx-lh-rostralmiddlefrontal',
    1028: 'ctx-lh-superiorfrontal',
    # 1029: 'ctx-lh-superiorparietal',
    1030: 'ctx-lh-superiortemporal',
    1031: 'ctx-lh-supramarginal',
    # 1034: 'ctx-lh-transversetemporal',
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
    # 2017: 'ctx-rh-paracentral',
    2018: 'ctx-rh-parsopercularis',
    2019: 'ctx-rh-parsorbitalis',
    2020: 'ctx-rh-parstriangularis',
    # 2021: 'ctx-rh-pericalcarine',
    2022: 'ctx-rh-postcentral',
    2023: 'ctx-rh-posteriorcingulate',
    2024: 'ctx-rh-precentral',
    2025: 'ctx-rh-precuneus',
    2026: 'ctx-rh-rostralanteriorcingulate',
    2027: 'ctx-rh-rostralmiddlefrontal',
    2028: 'ctx-rh-superiorfrontal',
    # 2029: 'ctx-rh-superiorparietal',
    2030: 'ctx-rh-superiortemporal',
    2031: 'ctx-rh-supramarginal',
    # 2034: 'ctx-rh-transversetemporal',
    2035: 'ctx-rh-insula'
}

ROI_SUBCORTEX = {
    10: 'Left-Thalamus',
    11: 'Left-Caudate',
    12: 'Left-Putamen',
    13: 'Left-Pallidum',
    17: 'Left-Hippocampus',
    18: 'Left-Amygdala',
    26: 'Left-Accumbens-area',
    28: 'Left-VentralDC',
    49: 'Right-Thalamus',
    50: 'Right-Caudate',
    51: 'Right-Putamen',
    52: 'Right-Pallidum',
    53: 'Right-Hippocampus',
    54: 'Right-Amygdala',
    58: 'Right-Accumbens-area',
    60: 'Right-VentralDC',
}

ROI_WM = {
    3001: "wm-lh-bankssts",
    3002: "wm-lh-caudalanteriorcingulate",
    3003: "wm-lh-caudalmiddlefrontal",
    3005: "wm-lh-cuneus",
    3006: "wm-lh-entorhinal",
    3007: "wm-lh-fusiform",
    3008: "wm-lh-inferiorparietal",
    3009: "wm-lh-inferiortemporal",
    3010: "wm-lh-isthmuscingulate",
    3011: "wm-lh-lateraloccipital",
    3012: "wm-lh-lateralorbitofrontal",
    3013: "wm-lh-lingual",
    3014: "wm-lh-medialorbitofrontal",
    3015: "wm-lh-middletemporal",
    3016: "wm-lh-parahippocampal",
    3018: "wm-lh-parsopercularis",
    3019: "wm-lh-parsorbitalis",
    3020: "wm-lh-parstriangularis",
    3022: "wm-lh-postcentral",
    3023: "wm-lh-posteriorcingulate",
    3024: "wm-lh-precentral",
    3025: "wm-lh-precuneus",
    3026: "wm-lh-rostralanteriorcingulat",
    3027: "wm-lh-rostralmiddlefrontal",
    3028: "wm-lh-superiorfrontal",
    3030: "wm-lh-superiortemporal",
    3031: "wm-lh-supramarginal",
    3032: "wm-lh-frontalpole",
    3033: "wm-lh-temporalpole",
    3035: "wm-lh-insula",
    4001: "wm-rh-bankssts",
    4002: "wm-rh-caudalanteriorcingulate",
    4003: "wm-rh-caudalmiddlefrontal",
    4005: "wm-rh-cuneus",
    4006: "wm-rh-entorhinal",
    4007: "wm-rh-fusiform",
    4008: "wm-rh-inferiorparietal",
    4009: "wm-rh-inferiortemporal",
    4010: "wm-rh-isthmuscingulate",
    4011: "wm-rh-lateraloccipital",
    4012: "wm-rh-lateralorbitofrontal",
    4013: "wm-rh-lingual ",
    4014: "wm-rh-medialorbitofrontal",
    4015: "wm-rh-middletemporal",
    4016: "wm-rh-parahippocampal",
    4018: "wm-rh-parsopercularis",
    4019: "wm-rh-parsorbitalis",
    4020: "wm-rh-parstriangularis",
    4022: "wm-rh-postcentral",
    4023: "wm-rh-posteriorcingulate",
    4024: "wm-rh-precentral",
    4025: "wm-rh-precuneus",
    4026: "wm-rh-rostralanteriorcingulat",
    4027: "wm-rh-rostralmiddlefrontal",
    4028: "wm-rh-superiorfrontal",
    4030: "wm-rh-superiortemporal",
    4031: "wm-rh-supramarginal",
    4032: "wm-rh-frontalpole",
    4033: "wm-rh-temporalpole",
    4035: "wm-rh-insula",
}


ROI_CORTEX_AND_GRAY_SUB_CORTEX = {
    1002: 'ctx-lh-caudalanteriorcingulate',
    1003: 'ctx-lh-caudalmiddlefrontal',
    1005: 'ctx-lh-cuneus',
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
    # 1017: 'ctx-lh-paracentral',
    1018: 'ctx-lh-parsopercularis',
    1019: 'ctx-lh-parsorbitalis',
    1020: 'ctx-lh-parstriangularis',
    # 1021: 'ctx-lh-pericalcarine',
    1022: 'ctx-lh-postcentral',
    1023: 'ctx-lh-posteriorcingulate',
    1024: 'ctx-lh-precentral',
    1025: 'ctx-lh-precuneus',
    1026: 'ctx-lh-rostralanteriorcingulate',
    1027: 'ctx-lh-rostralmiddlefrontal',
    1028: 'ctx-lh-superiorfrontal',
    # 1029: 'ctx-lh-superiorparietal',
    1030: 'ctx-lh-superiortemporal',
    1031: 'ctx-lh-supramarginal',
    # 1034: 'ctx-lh-transversetemporal',
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
    # 2017: 'ctx-rh-paracentral',
    2018: 'ctx-rh-parsopercularis',
    2019: 'ctx-rh-parsorbitalis',
    2020: 'ctx-rh-parstriangularis',
    # 2021: 'ctx-rh-pericalcarine',
    2022: 'ctx-rh-postcentral',
    2023: 'ctx-rh-posteriorcingulate',
    2024: 'ctx-rh-precentral',
    2025: 'ctx-rh-precuneus',
    2026: 'ctx-rh-rostralanteriorcingulate',
    2027: 'ctx-rh-rostralmiddlefrontal',
    2028: 'ctx-rh-superiorfrontal',
    # 2029: 'ctx-rh-superiorparietal',
    2030: 'ctx-rh-superiortemporal',
    2031: 'ctx-rh-supramarginal',
    # 2034: 'ctx-rh-transversetemporal',
    2035: 'ctx-rh-insula',
    10: 'Left-Thalamus',
    11: 'Left-Caudate',
    12: 'Left-Putamen',
    13: 'Left-Pallidum',
    17: 'Left-Hippocampus',
    18: 'Left-Amygdala',
    26: 'Left-Accumbens-area',
    28: 'Left-VentralDC',
    49: 'Right-Thalamus',
    50: 'Right-Caudate',
    51: 'Right-Putamen',
    52: 'Right-Pallidum',
    53: 'Right-Hippocampus',
    54: 'Right-Amygdala',
    58: 'Right-Accumbens-area',
    60: 'Right-VentralDC',
}

ROI_CORTEX_AND_WM_SUBCORTEX = {
    1002: 'ctx-lh-caudalanteriorcingulate',
    1003: 'ctx-lh-caudalmiddlefrontal',
    1005: 'ctx-lh-cuneus',
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
    # 1017: 'ctx-lh-paracentral',
    1018: 'ctx-lh-parsopercularis',
    1019: 'ctx-lh-parsorbitalis',
    1020: 'ctx-lh-parstriangularis',
    # 1021: 'ctx-lh-pericalcarine',
    1022: 'ctx-lh-postcentral',
    1023: 'ctx-lh-posteriorcingulate',
    1024: 'ctx-lh-precentral',
    1025: 'ctx-lh-precuneus',
    1026: 'ctx-lh-rostralanteriorcingulate',
    1027: 'ctx-lh-rostralmiddlefrontal',
    1028: 'ctx-lh-superiorfrontal',
    # 1029: 'ctx-lh-superiorparietal',
    1030: 'ctx-lh-superiortemporal',
    1031: 'ctx-lh-supramarginal',
    # 1034: 'ctx-lh-transversetemporal',
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
    # 2017: 'ctx-rh-paracentral',
    2018: 'ctx-rh-parsopercularis',
    2019: 'ctx-rh-parsorbitalis',
    2020: 'ctx-rh-parstriangularis',
    # 2021: 'ctx-rh-pericalcarine',
    2022: 'ctx-rh-postcentral',
    2023: 'ctx-rh-posteriorcingulate',
    2024: 'ctx-rh-precentral',
    2025: 'ctx-rh-precuneus',
    2026: 'ctx-rh-rostralanteriorcingulate',
    2027: 'ctx-rh-rostralmiddlefrontal',
    2028: 'ctx-rh-superiorfrontal',
    # 2029: 'ctx-rh-superiorparietal',
    2030: 'ctx-rh-superiortemporal',
    2031: 'ctx-rh-supramarginal',
    # 2034: 'ctx-rh-transversetemporal',
    2035: 'ctx-rh-insula',

    3001: "wm-lh-bankssts",
    3002: "wm-lh-caudalanteriorcingulate",
    3003: "wm-lh-caudalmiddlefrontal",
    3005: "wm-lh-cuneus",
    3006: "wm-lh-entorhinal",
    3007: "wm-lh-fusiform",
    3008: "wm-lh-inferiorparietal",
    3009: "wm-lh-inferiortemporal",
    3010: "wm-lh-isthmuscingulate",
    3011: "wm-lh-lateraloccipital",
    3012: "wm-lh-lateralorbitofrontal",
    3013: "wm-lh-lingual",
    3014: "wm-lh-medialorbitofrontal",
    3015: "wm-lh-middletemporal",
    3016: "wm-lh-parahippocampal",
    3018: "wm-lh-parsopercularis",
    3019: "wm-lh-parsorbitalis",
    3020: "wm-lh-parstriangularis",
    3022: "wm-lh-postcentral",
    3023: "wm-lh-posteriorcingulate",
    3024: "wm-lh-precentral",
    3025: "wm-lh-precuneus",
    3026: "wm-lh-rostralanteriorcingulat",
    3027: "wm-lh-rostralmiddlefrontal",
    3028: "wm-lh-superiorfrontal",
    3030: "wm-lh-superiortemporal",
    3031: "wm-lh-supramarginal",
    3032: "wm-lh-frontalpole",
    3033: "wm-lh-temporalpole",
    3035: "wm-lh-insula",
    4001: "wm-rh-bankssts",
    4002: "wm-rh-caudalanteriorcingulate",
    4003: "wm-rh-caudalmiddlefrontal",
    4005: "wm-rh-cuneus",
    4006: "wm-rh-entorhinal",
    4007: "wm-rh-fusiform",
    4008: "wm-rh-inferiorparietal",
    4009: "wm-rh-inferiortemporal",
    4010: "wm-rh-isthmuscingulate",
    4011: "wm-rh-lateraloccipital",
    4012: "wm-rh-lateralorbitofrontal",
    4013: "wm-rh-lingual ",
    4014: "wm-rh-medialorbitofrontal",
    4015: "wm-rh-middletemporal",
    4016: "wm-rh-parahippocampal",
    4018: "wm-rh-parsopercularis",
    4019: "wm-rh-parsorbitalis",
    4020: "wm-rh-parstriangularis",
    4022: "wm-rh-postcentral",
    4023: "wm-rh-posteriorcingulate",
    4024: "wm-rh-precentral",
    4025: "wm-rh-precuneus",
    4026: "wm-rh-rostralanteriorcingulat",
    4027: "wm-rh-rostralmiddlefrontal",
    4028: "wm-rh-superiorfrontal",
    4030: "wm-rh-superiortemporal",
    4031: "wm-rh-supramarginal",
    4032: "wm-rh-frontalpole",
    4033: "wm-rh-temporalpole",
    4035: "wm-rh-insula",
}

LOBES = {
    'frontal': ['ctx-lh-caudalmiddlefrontal', 'ctx-lh-lateralorbitofrontal', 'ctx-lh-medialorbitofrontal', 'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-parstriangularis', 'ctx-lh-precentral', 'ctx-lh-rostralmiddlefrontal', 'ctx-lh-superiorfrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-lateralorbitofrontal', 'ctx-rh-medialorbitofrontal', 'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-parstriangularis', 'ctx-rh-precentral', 'ctx-rh-rostralmiddlefrontal', 'ctx-rh-superiorfrontal', 'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate', 'ctx-lh-rostralanteriorcingulate', 'ctx-rh-rostralanteriorcingulate'],
    'parietal': ['ctx-lh-inferiorparietal', 'ctx-lh-superiorparietal', 'ctx-lh-postcentral', 'ctx-lh-precuneus', 'ctx-lh-supramarginal',  'ctx-rh-inferiorparietal', 'ctx-rh-superiorparietal', 'ctx-rh-postcentral', 'ctx-rh-precuneus', 'ctx-rh-supramarginal', 'ctx-lh-isthmuscingulate', 'ctx-rh-isthmuscingulate', 'ctx-lh-posteriorcingulate', 'ctx-rh-posteriorcingulate'],
    'temporal': ['ctx-lh-fusiform', 'ctx-lh-inferiortemporal', 'ctx-lh-middletemporal', 'ctx-lh-parahippocampal', 'ctx-lh-superiortemporal', 'ctx-lh-insula', 'ctx-rh-fusiform', 'ctx-rh-inferiortemporal', 'ctx-rh-middletemporal', 'ctx-rh-parahippocampal', 'ctx-rh-superiortemporal', 'ctx-rh-insula'],
    'occipital': ['ctx-lh-cuneus', 'ctx-lh-lateraloccipital', 'ctx-lh-lingual', 'ctx-lh-pericalcarine', 'ctx-rh-cuneus', 'ctx-rh-lateraloccipital', 'ctx-rh-lingual', 'ctx-rh-pericalcarine'],
}

BRAIN_SYSTEMS = {
    'Frontal': [
        'ctx-lh-caudalmiddlefrontal',
        'ctx-lh-lateralorbitofrontal',
        'ctx-lh-medialorbitofrontal',
        'ctx-lh-parsopercularis',
        'ctx-lh-parsorbitalis',
        'ctx-lh-parstriangularis',
        'ctx-lh-rostralmiddlefrontal',
        'ctx-lh-superiorfrontal',
        'ctx-rh-caudalmiddlefrontal',
        'ctx-rh-lateralorbitofrontal',
        'ctx-rh-medialorbitofrontal',
        'ctx-rh-parsopercularis',
        'ctx-rh-parsorbitalis',
        'ctx-rh-parstriangularis',
        'ctx-rh-rostralmiddlefrontal',
        'ctx-rh-superiorfrontal',
        'wm-lh-rostralmiddlefrontal',
        'wm-lh-superiorfrontal',
        'wm-lh-frontalpole',
        'wm-rh-caudalmiddlefrontal',
        'wm-rh-rostralmiddlefrontal',
        'wm-rh-superiorfrontal',
        'wm-rh-frontalpole',
        'wm-lh-caudalmiddlefrontal',
        'wm-lh-parstriangularis',
        'wm-rh-parstriangularis',
        'wm-lh-parsorbitalis',
        'wm-rh-parsorbitalis',
        'wm-lh-lateralorbitofrontal',     # Newly added
        'wm-rh-lateralorbitofrontal',     # Newly added
        'wm-lh-parsopercularis',          # Newly added
        'wm-rh-parsopercularis',          # Newly added
        'wm-lh-medialorbitofrontal',      # Newly added
        'wm-rh-medialorbitofrontal'       # Newly added
    ],
    'Parietal': [
        'ctx-lh-inferiorparietal',
        'ctx-lh-postcentral',
        'ctx-lh-precentral',
        'ctx-lh-supramarginal',
        'ctx-rh-inferiorparietal',
        'ctx-rh-postcentral',
        'ctx-rh-precentral',
        'ctx-rh-supramarginal',
        'wm-lh-inferiorparietal',
        'wm-lh-postcentral',
        'wm-lh-precentral',
        'wm-lh-supramarginal',
        'wm-rh-inferiorparietal',
        'wm-rh-postcentral',
        'wm-rh-precentral',
        'wm-rh-supramarginal'
    ], 
    'Occipital': [
        'ctx-lh-cuneus',
        'ctx-lh-lateraloccipital',
        'ctx-rh-lateraloccipital',
        'wm-lh-cuneus',
        'wm-lh-lateraloccipital',
        'wm-rh-cuneus',
        'wm-rh-lateraloccipital',
        'ctx-rh-cuneus'                   # Newly added
    ],
    'Temporal': [
        'ctx-lh-fusiform',
        'ctx-lh-inferiortemporal',
        'ctx-lh-lingual',
        'ctx-lh-middletemporal',
        'ctx-lh-parahippocampal',
        'ctx-lh-superiortemporal',
        'ctx-rh-fusiform',
        'ctx-rh-inferiortemporal',
        'ctx-rh-lingual',
        'ctx-rh-middletemporal',
        'ctx-rh-parahippocampal',
        'ctx-rh-superiortemporal',
        'wm-lh-fusiform',
        'wm-lh-inferiortemporal',
        'wm-lh-lingual',
        'wm-lh-middletemporal',
        'wm-lh-parahippocampal',
        'wm-lh-superiortemporal',
        'wm-rh-fusiform',
        'wm-rh-inferiortemporal',
        'wm-rh-lingual',
        'wm-rh-middletemporal',
        'wm-rh-parahippocampal',
        'wm-rh-superiortemporal',
        'wm-lh-entorhinal',
        'wm-rh-entorhinal',
        'wm-rh-lingual ',  # Newly added (space included to match exactly),
        "wm-rh-bankssts",
        "wm-lh-bankssts",
        "wm-lh-temporalpole",
        "wm-rh-temporalpole"
    ],
    'Limbic': [
        'ctx-lh-caudalanteriorcingulate',
        'ctx-lh-isthmuscingulate',
        'ctx-lh-posteriorcingulate',
        'ctx-lh-precuneus',
        'ctx-lh-rostralanteriorcingulate',
        'ctx-lh-insula',
        'ctx-rh-caudalanteriorcingulate',
        'ctx-rh-isthmuscingulate',
        'ctx-rh-posteriorcingulate',
        'ctx-rh-precuneus',
        'ctx-rh-rostralanteriorcingulate',
        'ctx-rh-insula',
        'Left-Hippocampus',
        'Left-Amygdala',
        'Right-Hippocampus',
        'Right-Amygdala',
        'wm-lh-caudalanteriorcingulate',
        'wm-lh-isthmuscingulate',
        'wm-lh-posteriorcingulate',
        'wm-lh-precuneus',
        'wm-lh-rostralanteriorcingulat',
        'wm-lh-insula',
        'wm-rh-caudalanteriorcingulate',
        'wm-rh-isthmuscingulate',
        'wm-rh-posteriorcingulate',
        'wm-rh-precuneus',
        'wm-rh-rostralanteriorcingulat',
        'wm-rh-insula'
    ],
    'Basal': [
        'Left-Caudate',
        'Left-Putamen',
        'Left-Pallidum',
        'Left-Accumbens-area',
        'Right-Caudate',
        'Right-Putamen',
        'Right-Pallidum',
        'Right-Accumbens-area'
    ],
    'Thalamus': [
        'Left-Thalamus',
        'Left-VentralDC',
        'Right-Thalamus',
        'Right-VentralDC'
    ]
}


# Map brain systems to colors
BRAIN_SYSTEMS_COLORS = {
    'Frontal': 'red',
    'Parietal': 'blue',
    'Occipital': 'green',
    'Temporal': 'purple',
    'Limbic': 'orange',
    'Basal': 'brown',
    'Thalamus': 'cyan'
}

DICT_NUM_TO_ROIS = {1: SUB_CORTEX_DICT,
                    2: ROI_PUTAMEN_THALAMUS,
                    3: ROI_PALLIDUM_PUTAMEN_CAUDETE,
                    4: ROI_PUTAMEN_CAUDETE,
                    5: ROI_AMYGDALA_HIPPOCAMPUS,
                    6: ROI_ACCUM_HIPPO_AMYG}

# -------------------- MRI Physical Parameters -------------------- #
BASIC_4_PARAMS = ["r1", "tv", "r2s", "mt"]
BASIC_4_PARAMS_WITH_SLOPES = ["r1", "tv", "r2s", "mt",
                              "Slope-tv-r1", "Slope-tv-r2s", "Dtv-r1-values", "Dtv-r2s-values"]
ALL_PARAMS = ["r1", "tv", "r2s", "mt", "t2", "diffusion_fa", "diffusion_md"]
ALL_PARAMS_WITH_SLOPES = ["r1", "tv", "r2s", "mt", "t2", "diffusion_fa", "diffusion_md",
                          "Slope-tv-r1", "Slope-tv-r2s", "Slope-tv-mt", "Slope-tv-t2",
                          "Slope-tv-diffusion_fa", "Slope-tv-diffusion_md", "Slope-r2s-r1"]
SLOPES = ["Slope-tv-r1", "Slope-tv-r2s", "Slope-tv-mt", "Slope-tv-t2",
          "Slope-tv-diffusion_fa", "Slope-tv-diffusion_md", "Slope-r2s-r1"]

PARAMETERS = ["r1", "tv", "r2s", "mt", "t2", "diffusion"]
PARAMETERS_W_D_TV_R1_AND_R2S = ["r1", "tv", "r2s", "mt", "t2", "diffusion", "Slope-tv-r1", "Slope-tv-r2s",
                                "Dtv-r1-values", "Dtv-r2s-values"]
PARAMS_OF_SLOPES = ["Slope-tv-r1", "Slope-tv-r2s",
                    "Dtv-r1-values", "Dtv-r2s-values"]

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

BASIC_SEG_WM = os.getenv('BASIC_SEG_WM')
SEG_T2_WM = os.getenv('SEG_T2_WM')
SEG_DIFFUSION_WM = os.getenv('SEG_DIFFUSION_WM')

# Plots values
COLOR_LIST = {
    0: 3,
    1: 1016,
    2: 1024,
    3: 26
}

EXAMPLE_ANNOT_LH_PATH = os.getenv('EXAMPLE_ANNOT_LH_PATH')
EXAMPLE_ANNOT_RH_PATH = os.getenv('EXAMPLE_ANNOT_RH_PATH')
EXAMPLE_SURFACE_PIAL_LH_PATH = os.getenv('EXAMPLE_SURFACE_PIAL_LH_PATH')
EXAMPLE_SURFACE_PIAL_RH_PATH = os.getenv('EXAMPLE_SURFACE_PIAL_RH_PATH')

WANDB_ENTITY = os.getenv('WANDB_ENTITY')
