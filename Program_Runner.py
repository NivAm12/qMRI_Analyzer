import enum
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from Statistics import StatisticsWrapper
from Data_Processor import DataProcessor
from sklearn.preprocessing import RobustScaler
import pickle
sns.set_theme(style="ticks", color_codes=True)

# -------------------- PATHS -------------------- #
from constants import PATH_TO_RAW_DATA,PATH_TO_RAW_DATA_6_PARAMS, SAVE_DATA_PATH, SUBJECTS_INFO_PATH, SAVE_DATA_OF_ALL_Z_SCORE_MEANS, \
    SAVE_DATA_OF_ADULTS_Z_SCORE_MEANS, SAVE_DATA_OF_YOUNG_Z_SCORE_MEANS, PATH_TO_RAW_DATA_Z_SCORED, \
    PATH_TO_RAW_DATA_ROBUST_SCALED

# -------------------- File Names -------------------- #
from constants import HIERARCHICAL_CLUSTERING_FILE

# -------------------- Folders ---------------------- #
from constants import Z_SCORE_ON_AVG_ON_BRAIN_DIR, Z_SCORE_ON_BRAIN_DIR, NORMALIZE_BY_MEDIAN_DIR, \
    MEANS_ON_BRAIN_DIR, STD_OF_PARAMS_BRAIN_DIR

# -------------------- ROIs -------------------- #
from constants import SUB_CORTEX_DICT, ROI_PUTAMEN_CAUDETE, ROI_PUTAMEN_THALAMUS, \
    ROI_AMYGDALA_HIPPOCAMPUS, ROI_PALLIDUM_PUTAMEN_CAUDETE, ROI_ACCUM_HIPPO_AMYG, DICT_NUM_TO_ROIS

# -------------- Sub Folders - by ROIS ------------- #
from constants import PUTAMEN_CAUDETE_DIR, PALLIDUM_PUTAMEN_CAUDETE_DIR, AMYGDALA_HIPPOCAMPUS_DIR

# -------------- Sub Folders - by Raw Data Type ------------- #
from constants import RAW_DATA_DIR, RAW_DATA_ROBUST_SCALED_DIR, RAW_DATA_Z_SCORED_DIR

# -------------- Type of Raw Data ------------- #
from constants import RAW_DATA, Z_SCORE, ROBUST_SCALING, RAW_DATA_6_PARAMS, DICT_NUM_TO_METHOD

# -------------------- MRI Physical Parameters -------------------- #
from constants import BASIC_4_PARAMS, PARAMETERS, PARAMETERS_W_D_TV_R1_AND_R2S, PARAMS_OF_SLOPES

# -------------------- Magic Number -------------------- #
from constants import OLD, YOUNG, AGE_THRESHOLD

# -------------------- Statistics Funcs to Run -------------------- #
from constants import T_TEST, HIERARCHICAL_CLUSTERING_WITH_CORRELATIONS, SD_PER_PARAMETER, PLOT_DATA_PER_PARAM, \
    PLOT_DATA_PER_ROI_PER_SUBJET_WITH_ALL_PARAMS

# ------------------- Statistics funcs on processed data - adding normalizer/means ---------------------- #
from Statistics import StatisticsWrapper


# -------------------- Enums for statistical actions -------------------- #
class Actions(enum.Enum):
    z_score = 1  # Z Score on data - per subject, per parameter, per ROI
    z_score_means = 2  # Z Score on means of all subjects, per parameters, per its ROI
    means_per_subject = 3  # Means on each Subject, per its ROI (per parameter)
    robust_scaling = 4  # subtracting the median and dividing by the interquartile range


# -------------- Dictionaries of raw_data_type to their input path and output Dir ------------- #
RAW_DATA_NORMALIZER_PATH = {RAW_DATA: PATH_TO_RAW_DATA, Z_SCORE: PATH_TO_RAW_DATA_Z_SCORED,
                            ROBUST_SCALING: PATH_TO_RAW_DATA_ROBUST_SCALED, RAW_DATA_6_PARAMS: PATH_TO_RAW_DATA_6_PARAMS}

RAW_DATA_NORMALIZER_OUTPUT_DIR = {RAW_DATA: RAW_DATA_DIR, Z_SCORE: RAW_DATA_Z_SCORED_DIR,
                                  ROBUST_SCALING: RAW_DATA_ROBUST_SCALED_DIR, RAW_DATA_6_PARAMS: RAW_DATA_DIR}

# -------------------- Dict action:function -------------------- #
ACTION_FUNCTION_DICT = {Actions.z_score: StatisticsWrapper.calc_z_score_per_subject,
                        Actions.z_score_means: StatisticsWrapper.calc_z_score_on_mean_per_subject2,
                        Actions.means_per_subject: StatisticsWrapper.calc_mean_per_subject_per_parameter_per_ROI,
                        Actions.robust_scaling: StatisticsWrapper.calc_mean_robust_scaling_per_subject_per_parameter_per_ROI}

ACTION_SAVE_ADDRESS_DICT = {Actions.z_score: Z_SCORE_ON_BRAIN_DIR,
                            Actions.z_score_means: Z_SCORE_ON_AVG_ON_BRAIN_DIR,
                            Actions.means_per_subject: MEANS_ON_BRAIN_DIR,
                            Actions.robust_scaling: NORMALIZE_BY_MEDIAN_DIR}


def get_save_address(output_path, raw_data_type, manipulation_on_data_name, dir_name):
    """
    Uses default save data path (which is "/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/Covariance_Aging/
    saved_versions/corr_by_means/subcortical_updated/with_R1/Analysis/young_adults_comparison/)
    and adds to it the raw_data_type, which is either raw, z_scored or robust_scaling.
    :param output_path: Whole output path
    :param raw_data_type: the raw_data_type, which is either raw, z_scored or robust_scaling
    :param manipulation_on_data_name: name of normalizing/other manipulation done on the data
    :param dir_name: name of the dir it will be added to.
    :return:the updated path for saving the files
    """
    return output_path + raw_data_type + manipulation_on_data_name + dir_name


def analyse_data(subjects_raw_data, statistics_func, save_address, funcs_to_run, params_to_work_with,
                 ROIs_to_analyze=SUB_CORTEX_DICT):
    """
    This is the analyzing part -
        1) Take the raw data (raw, z_scored raw data or robust scaling on the raw data) -> and make manipulation
           usually if its raw we use a normalizer (robust scaling/z_score) and take the average for each roi per
           parameter, and if the raw data is already normalized we just take the average (RECOMMENDED).
        2) The raw data is diverged by a certain parameter - a column and a threshold, for example -> column 'age' and
            it threshold - 40.
        3) Each of chosen function is activated -> correlations, t-tests, sd etc'
    :param subjects_raw_data: The subject's raw data:
                    subject | ROI  |   r1  |   r2s    |    mt    |    tv  | diffusion | t2 |
                    H20_AS  |  10 |  list[values of voxels]
                    H20_AS  |  11 |  list[values of voxels]
    :param statistics_func: the statistics func that suppose to normalize the data / make it means / other manipulation
    :param save_address: the saving address path for the outputs.
    :param funcs_to_run: all the statistics funcs to run on the data.
    :param ROIs_to_analyze: all the rois to analyze the data with.
    :param params_to_work_with: all the parameters (r1, r2s, mt, tv, diffusion, t2 etc) to analyze data with
    :return: None
    """
    analyzed_data = statistics_func(subjects_raw_data, params_to_work_with)
    chosen_data = StatisticsWrapper.chose_relevant_data(analyzed_data, ROIs_to_analyze, params_to_work_with)

    # You can choose here which column you want ('Age' / 'Gender' / etc') and the threshold (AGE_THRESHOLD / 'M' / etc')
    # and the names of each group.
    group_a_name, group_b_name, col_divider, threshold = YOUNG, OLD, 'Age', AGE_THRESHOLD

    young_subjects, old_subjects = StatisticsWrapper.seperate_data_to_two_groups(chosen_data, col_divider, threshold)
    for func in funcs_to_run:
        if func == T_TEST:
            StatisticsWrapper.t_test_per_parameter_per_area(young_subjects, old_subjects, ROIs_to_analyze,
                                                            'ROI', group_a_name, group_b_name)

        elif func == PLOT_DATA_PER_PARAM:
            StatisticsWrapper.plot_data_per_param_per_roi_next_to_each_other(young_subjects, old_subjects,
                                                                             group_a_name, group_b_name, save_address)

        elif func == SD_PER_PARAMETER:
            StatisticsWrapper.computed_std_per_parameter(young_subjects, old_subjects, params_to_work_with,
                                                         list(ROIs_to_analyze.keys()), group_a_name, group_b_name,
                                                         save_address, visualize=True)

        elif func == HIERARCHICAL_CLUSTERING_WITH_CORRELATIONS:
            StatisticsWrapper.calculate_correlation_per_data(chosen_data, params_to_work_with, ROIs_to_analyze, "ALL",
                                                             save_address)
            StatisticsWrapper.calculate_correlation_per_data(young_subjects, params_to_work_with, ROIs_to_analyze,
                                                             group_a_name, save_address + "/" + group_a_name + "/")
            StatisticsWrapper.calculate_correlation_per_data(old_subjects, params_to_work_with, ROIs_to_analyze,
                                                             group_b_name, save_address + "/" + group_b_name + "/")


        elif func == PLOT_DATA_PER_ROI_PER_SUBJET_WITH_ALL_PARAMS:
            StatisticsWrapper.plot_values_per_parameter_per_roi(chosen_data, params_to_work_with, list(ROIs_to_analyze),
                                                                save_address)


def run_program(pattern, raw_data_path, save_address, funcs_to_run, chosen_rois_dict, params_to_work_with):
    """
    Run Program
    :param pattern: the pattern we chose (how to manipulate the data)
    :param raw_data_path: the path to the raw data
    :param save_address: the saving address - output path
    :param funcs_to_run: all the functions to run
    :param chosen_rois_dict: all the chosen rois
    :param params_to_work_with: all the parameters to work with.
    :return: None
    """
    # Process Data
    subjects_raw_data = DataProcessor(raw_data_path, SUB_CORTEX_DICT, chosen_rois_dict).get_data_proccessed()

    # Choose Statistics
    statistics_func = ACTION_FUNCTION_DICT[pattern]

    # Analyse data
    analyse_data(subjects_raw_data, statistics_func, save_address, funcs_to_run, params_to_work_with, chosen_rois_dict)


def run():
    """
    For programmer -> Here you CAN/SHOULD choose how to run the program:

        1. pattern: choose the pattern of the action you would like to do on the raw data.
        2. raw_data_type: choose kind of raw data you would like to get - z-score/robust-scaling/not normalized
                    Recommended: Use the RAW DATA that was ALREADY normalized like z-score/robust scaling (since it has
                    been normalized on the whole brain), and than choose to calculate the means!
                    (pattern = Actions.means_per_subject)!
                    other manipulation exists but may not be as useful/accurate.
        3. chosen_rois_dict: choose the dictionary containing all the relevant info. Make sure the raw data contains all
                            the relevant ROIs
        4. funcs_to_run: list of funcs name (constants) the you can run. Choose form the constants.py file.
        5. params_to_work_with: Choose the params you want to work with (r1, r2s, diffusion, mt, tv, t2, etc') - look
                                at the constant.py for all the options.
        6. output_path: choose output path (default is SAVE_DATA_PATH)
        7. rois_output_dir: if you would like to add a directory that it's name will be the rois/project you work on

        8. save_address: Possible to change, not recommended

    PLEASE NOTE:
        (A) Most of the changes here just require you to look on the relevant place in the constants.py file
        - for example to look for other dictionaries of rois just open the constants.py file and look for those kind
        of dictionaries.
        (B) If you added another function in Statistics.py / added more types of raw data / etc you should:
            I. Add a const of the function name / path to constans.py
            II. Add another elif in analyse_data func, with the function name (depends if you added function)
                OR
                Add RAW_DATA_NORMALIZER_PATH another path to the new raw_data_type
                OR
                Add to Actions another func
            III. Change HERE the relevant variable (add the const to the list of funcs / change raw_data_type)

    :return: None
    """
    # Change here the action to go on the raw data (look at actions options)
    pattern = Actions.means_per_subject

    # Change here the type of raw data you would like (RAW_DATA, Z_SCORE, ROBUST_SCALING)
    raw_data_type = Z_SCORE

    # DONT CHANGE - from here get the raw data
    raw_data_path = RAW_DATA_NORMALIZER_PATH[raw_data_type]

    # Change Here the rois you would like to work with
    chosen_rois_dict = SUB_CORTEX_DICT

    # Change here the Statistics funcs to run
    funcs_to_run = [PLOT_DATA_PER_ROI_PER_SUBJET_WITH_ALL_PARAMS, T_TEST, PLOT_DATA_PER_PARAM,
                    SD_PER_PARAMETER, HIERARCHICAL_CLUSTERING_WITH_CORRELATIONS]

    # Choose here the parameters to work with in the data
    params_to_work_with = PARAMETERS

    # Change here the path to save the results to - default is SAVE_DATA_PATH:
    output_path = SAVE_DATA_PATH

    # If you chose another ROIs - you can put them in another sub-dir inside the raw_data_type dir :)
    rois_output_dir = ""

    # DONT CHANGE - from here you get the directory name in which the file will be saved
    raw_output_data_dir = RAW_DATA_NORMALIZER_OUTPUT_DIR[raw_data_type]


    # Possible to change Save Address: the format is as following :
    # <output_path>/<dir_name_by_raw_data_type>/<dir name by manipulation done on the data-like z_score>/<possible - ROIs took place in the run>
    save_address = get_save_address(output_path, raw_output_data_dir, ACTION_SAVE_ADDRESS_DICT[pattern],
                                    rois_output_dir)

    # Run the Program
    run_program(pattern, raw_data_path, save_address, funcs_to_run, chosen_rois_dict, params_to_work_with)


def main():
    """
    The main function which run the program -> here you can chose in which mode to run it:
    user -> will need to insert some information by your own and chose options.
    programmer -> you can change it manually be choosing from constants
    :return:
    """
    run()


if __name__ == "__main__":
    main()
