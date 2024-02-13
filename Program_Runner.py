import enum
import seaborn as sns
from data_handling.Data_Processor import DataProcessor
import constants
import os
from statistics_methods.Statistics import StatisticsWrapper

sns.set_theme(style="ticks", color_codes=True)


# -------------------- Enums for statistical actions -------------------- #
class Actions(enum.Enum):
    z_score = 1  # Z Score on data - per subject, per parameter, per ROI
    z_score_means = 2  # Z Score on means of all subjects, per parameters, per its ROI
    median_per_subject = 3  # Median on each Subject, per its ROI (per parameter)
    robust_scaling = 4  # subtracting the median and dividing by the interquartile range


RAW_DATA_NORMALIZER_OUTPUT_DIR = {constants.RAW_DATA: constants.RAW_DATA_DIR,
                                  constants.Z_SCORE: constants.RAW_DATA_Z_SCORED_DIR,
                                  constants.ROBUST_SCALING: constants.RAW_DATA_ROBUST_SCALED_DIR,
                                  constants.RAW_DATA_6_PARAMS: constants.RAW_DATA_DIR}

ACTION_FUNCTION_DICT = {Actions.z_score: StatisticsWrapper.calc_z_score_per_subject,
                        Actions.z_score_means: StatisticsWrapper.calc_z_score_on_mean_per_subject2,
                        Actions.median_per_subject: StatisticsWrapper.calc_mean_per_subject_per_parameter_per_ROI,
                        Actions.robust_scaling: StatisticsWrapper.calc_mean_robust_scaling_per_subject_per_parameter_per_ROI}

ACTION_SAVE_ADDRESS_DICT = {Actions.z_score: constants.Z_SCORE_ON_BRAIN_DIR,
                            Actions.z_score_means: constants.Z_SCORE_ON_AVG_ON_BRAIN_DIR,
                            Actions.median_per_subject: constants.MEANS_ON_BRAIN_DIR,
                            Actions.robust_scaling: constants.NORMALIZE_BY_MEDIAN_DIR}


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
                 ROIs_to_analyze, raw_params, project_name):
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
    chosen_data = StatisticsWrapper.chose_relevant_data(analyzed_data, ROIs_to_analyze, raw_params, params_to_work_with)

    # You can choose here which column you want ('Age' / 'Gender' / etc') and the threshold (AGE_THRESHOLD / 'M' / etc')
    # and the names of each group.
    group_a_name, group_b_name, col_divider, threshold = constants.YOUNG, constants.OLD, 'Age', constants.AGE_THRESHOLD
    young_subjects, old_subjects = StatisticsWrapper.seperate_data_to_two_groups(chosen_data, col_divider, threshold)

    for func in funcs_to_run:
        if func == constants.PLOT_DATA_PER_PARAM:
            StatisticsWrapper.plot_data_per_param_per_roi_next_to_each_other(young_subjects, old_subjects,
                                                                             params_to_work_with,
                                                                             group_a_name, group_b_name,
                                                                             save_address, project_name)

        elif func == constants.SD_PER_PARAMETER:
            StatisticsWrapper.computed_std_per_parameter(young_subjects, old_subjects, params_to_work_with,
                                                         list(ROIs_to_analyze.keys()), group_a_name, group_b_name,
                                                         save_address,
                                                         project_name=project_name)

        elif func == constants.HIERARCHICAL_CLUSTERING_WITH_CORRELATIONS:
            StatisticsWrapper.calculate_correlation_per_data(chosen_data, params_to_work_with, ROIs_to_analyze, "ALL",
                                                             save_address)
            # StatisticsWrapper.calculate_correlation_per_data(young_subjects, params_to_work_with, ROIs_to_analyze,
            #                                                  group_a_name, save_address + "/" + group_a_name + "/",
            #                                                  project_name=project_name)
            # StatisticsWrapper.calculate_correlation_per_data(old_subjects, params_to_work_with, ROIs_to_analyze,
            #                                                  group_b_name, save_address + "/" + group_b_name + "/",
            #                                                  project_name=project_name)

        elif func == constants.HIERARCHICAL_CLUSTERING:
            # for linkage_metric in constants.LINKAGE_METRICS:
            #     StatisticsWrapper.hierarchical_clustering(chosen_data, params_to_work_with, linkage_metric,
            #                                               project_name, "all")
            #     StatisticsWrapper.hierarchical_clustering(young_subjects, params_to_work_with, linkage_metric,
            #                                               project_name, group_a_name)
            #     StatisticsWrapper.hierarchical_clustering(old_subjects, params_to_work_with, linkage_metric,
            #                                               project_name, group_b_name)
            StatisticsWrapper.hierarchical_clustering(chosen_data, params_to_work_with, 'complete',
                                                      project_name, "all")
            StatisticsWrapper.hierarchical_clustering(young_subjects, params_to_work_with, 'complete',
                                                      project_name, group_a_name)
            StatisticsWrapper.hierarchical_clustering(old_subjects, params_to_work_with, 'complete',
                                                      project_name, group_b_name)

        elif func == constants.ROIS_CORRELATIONS:
            clusters_rois = StatisticsWrapper.hierarchical_clustering(chosen_data, params_to_work_with, 'complete',
                                                                      title="all")['dendrogram_data']['ivl']
            young_result = StatisticsWrapper.roi_correlations(young_subjects, params_to_work_with, clusters_rois,
                                                              'young',
                                                              project_name)
            old_result = StatisticsWrapper.roi_correlations(old_subjects, params_to_work_with, clusters_rois, 'old',
                                                            project_name)

            StatisticsWrapper.plot_heatmap(old_result - young_result, 'differences of old and young', project_name)

        elif func == constants.PLOT_BRAIN_CLUSTERS:
            # for linkage_metric in constants.LINKAGE_METRICS:
            linkage_metric = 'complete'
            young_dendrogram_data = \
                StatisticsWrapper.hierarchical_clustering(young_subjects, params_to_work_with,
                                                          linkage_metric=linkage_metric,
                                                          title="young")

            old_dendrogram_data = \
                StatisticsWrapper.hierarchical_clustering(old_subjects, params_to_work_with,
                                                          linkage_metric=linkage_metric,
                                                          title="old")

            StatisticsWrapper.plot_clusters_on_brain(young_dendrogram_data['clusters'], chosen_data.iloc[0].subjects,
                                                     chosen_rois_dict, distance_to_cluster=6,
                                                     title=f'young_with_{linkage_metric}', project_name=project_name)
            StatisticsWrapper.plot_clusters_on_brain(old_dendrogram_data['clusters'], chosen_data.iloc[0].subjects,
                                                     chosen_rois_dict, distance_to_cluster=6,
                                                     title=f'old_with_{linkage_metric}', project_name=project_name)


def run_program(pattern, raw_data_path, save_address, funcs_to_run, chosen_rois_dict, params_to_work_with,
                raw_params, project_name):
    """
    Run Program
    :param pattern: the pattern we chose (how to manipulate the data)
    :param raw_data_path: the path to the raw data
    :param save_address: the saving address - output path
    :param funcs_to_run: all the functions to run
    :param chosen_rois_dict: all the chosen rois
    :param params_to_work_with: all the parameters to work with.
    :param raw_params: original params of the data
    :param project_name: wandb project name
    :return: None
    """
    # Process Data
    subjects_raw_data = DataProcessor(raw_data_path, chosen_rois_dict, chosen_rois_dict).get_data_proccessed()

    # Choose Statistics
    statistics_func = ACTION_FUNCTION_DICT[pattern]

    # Analyse data
    analyse_data(subjects_raw_data, statistics_func, save_address, funcs_to_run, params_to_work_with, chosen_rois_dict,
                 raw_params, project_name)


if __name__ == "__main__":
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
    pattern = Actions.median_per_subject

    # Change here the type of raw data you would like (RAW_DATA, Z_SCORE, ROBUST_SCALING)
    raw_data_type = constants.Z_SCORE

    # get the raw data
    raw_data_path = constants.PATH_TO_CORTEX_all_params_z_score

    # Change Here the rois you would like to work with
    chosen_rois_dict = constants.ROI_CORTEX

    # wandb
    # project_name = 'CORTEX_4_params_no_slopes_38_subjects'
    project_name = None

    # Change here the Statistics funcs to run
    funcs_to_run = [constants.HIERARCHICAL_CLUSTERING]

    # Choose here the parameters to work with in the data
    data_params = constants.ALL_PARAMS_WITH_SLOPES
    params_to_work_with = constants.ALL_PARAMS_WITH_SLOPES

    # Change here the path to save the results to - default is SAVE_DATA_PATH:
    output_path = constants.SAVE_DATA_PATH

    # If you chose another ROIs - you can put them in another sub-dir inside the raw_data_type dir :)
    rois_output_dir = ""

    # DON'T CHANGE - from here you get the directory name in which the file will be saved
    raw_output_data_dir = RAW_DATA_NORMALIZER_OUTPUT_DIR[raw_data_type]

    # Possible to change Save Address: the format is as following :
    save_address = get_save_address(output_path, raw_output_data_dir, ACTION_SAVE_ADDRESS_DICT[pattern],
                                    rois_output_dir)

    # Run the Program
    os.environ['WANDB_ENTITY'] = constants.WANDB_ENTITY
    run_program(pattern, raw_data_path, save_address, funcs_to_run, chosen_rois_dict, params_to_work_with, data_params,
                project_name)
