import os
import enum
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Any, Tuple
import wandb


# -------------------- PATHS -------------------- #
from constants import PATH_TO_RAW_DATA, SAVE_DATA_PATH, SUBJECTS_INFO_PATH, SAVE_DATA_OF_ALL_Z_SCORE_MEANS, \
    SAVE_DATA_OF_ADULTS_Z_SCORE_MEANS, SAVE_DATA_OF_YOUNG_Z_SCORE_MEANS, PATH_TO_RAW_DATA_Z_SCORED, \
    PATH_TO_RAW_DATA_ROBUST_SCALED

# -------------------- File Names -------------------- #
from constants import HIERARCHICAL_CLUSTERING_FILE

# -------------------- Folders ---------------------- #
from constants import Z_SCORE_ON_AVG_ON_BRAIN_DIR, Z_SCORE_ON_BRAIN_DIR, NORMALIZE_BY_MEDIAN_DIR, \
    MEANS_ON_BRAIN_DIR, STD_OF_PARAMS_BRAIN_DIR

# -------------------- ROIs -------------------- #
from constants import SUB_CORTEX_DICT, ROI_PUTAMEN_CAUDETE, ROI_PUTAMEN_THALAMUS, \
    ROI_AMYGDALA_HIPPOCAMPUS, ROI_PALLIDUM_PUTAMEN_CAUDETE, ROI_ACCUM_HIPPO_AMYG

# -------------- Sub Folders - by ROIS ------------- #
from constants import PUTAMEN_CAUDETE_DIR, PALLIDUM_PUTAMEN_CAUDETE_DIR, AMYGDALA_HIPPOCAMPUS_DIR

# -------------- Sub Folders - by Raw Data Type ------------- #
from constants import RAW_DATA_DIR, RAW_DATA_ROBUST_SCALED_DIR, RAW_DATA_Z_SCORED_DIR

# -------------- Type of Raw Data ------------- #
from constants import RAW_DATA, Z_SCORE, ROBUST_SCALING

# -------------------- MRI Physical Parameters -------------------- #
from constants import PARAMETERS, PARAMETERS_W_D_TV_R1_AND_R2S

# -------------------- Magic Number -------------------- #
from constants import OLD, YOUNG, AGE_THRESHOLD


# -------------------- Enums for statistical actions -------------------- #
class Actions(enum.Enum):
    z_score = 1  # Z Score on data - per subject, per parameter, per ROI
    z_score_means = 2  # Z Score on means of all subjects, per parameters, per its ROI
    means_per_subject = 3  # Means on each Subject, per its ROI (per parameter)
    robust_scaling = 4  # subtracting the median and dividing by the interquartile range


# -------------- Dictionaries of raw_data_type to their input path and output Dir ------------- #
RAW_DATA_NORMALIZER_PATH = {RAW_DATA: PATH_TO_RAW_DATA, Z_SCORE: PATH_TO_RAW_DATA_Z_SCORED,
                            ROBUST_SCALING: PATH_TO_RAW_DATA_ROBUST_SCALED}

RAW_DATA_NORMALIZER_OUTPUT_DIR = {RAW_DATA: RAW_DATA_DIR, Z_SCORE: RAW_DATA_Z_SCORED_DIR,
                                  ROBUST_SCALING: RAW_DATA_ROBUST_SCALED_DIR}

sns.set_theme(style="ticks", color_codes=True)


class StatisticsWrapper:
    """
    This static class makes statistical analysis on the data, for example:
    - Normalizing the data using z-score / robust scaling
    - Calculates T-Test between each parameter in each ROI between two groups.
    - Standard deviation in each ROI per parameter.
    - Distribution in each ROI per parameter.
    - Hierarchical Clustering + correlations between each ROI between subject and the average of them.
    """

    @staticmethod
    def computed_zscore_per_values(cell):
        """
        Computed z_score per cell
        :param cell: current cell in the dataframe (suppose to be an array)
        :return: return the array after normalizing it by z_score
        """
        return stats.zscore(np.array(cell))

    @staticmethod
    def convert_to_zscore_by_means(cell):
        """
        Takes all means from each ROI in the cell, concat them, does z-score on all of them and return the
        result after z_score

        :param cell:
        :return:
        """
        new_cell = []
        for key in cell:
            new_cell += [cell[key]]
        new_cell = stats.zscore(np.array(new_cell))
        i = 0
        for key in cell:
            cell[key] = new_cell[i]
            i += 1
        return new_cell

    @staticmethod
    def calc_std_per_subject_per_parameter_per_ROI(subjects_raw_data, params):
        """
        calculates SD per subject per parameter per ROI.
        :param subjects_raw_data: given data
        :param params: given relevant params to analyze.
        :return: the data after being manipulated to SD
        """
        std_per_subject_per_roi_per_param = subjects_raw_data.copy()
        std_per_subject_per_roi_per_param[params] = std_per_subject_per_roi_per_param[params].applymap(np.std)
        return std_per_subject_per_roi_per_param

    @staticmethod
    def calc_mean_per_subject_per_parameter_per_ROI(subjects_raw_data, params):
        """
        calculates means per subject per parameter per ROI.
        :param subjects_raw_data: given data
        :param params: given relevant params to analyze.
        :return: the data after being manipulated to means
        """
        mean_per_subject_per_roi_per_param = subjects_raw_data.copy()
        mean_per_subject_per_roi_per_param[params] = mean_per_subject_per_roi_per_param[params].applymap(np.mean)
        return mean_per_subject_per_roi_per_param

    @staticmethod
    def calc_z_score_per_subject(subjects_raw_data, params):
        """
        Calculated z-score per subject by given raw data and params
        :param subjects_raw_data: given data
        :param params: given relevant params to analyze.
        :return: the data after being manipulated to z-score
        """
        zscore_per_subject_per_roi_per_param = subjects_raw_data.copy()
        zscore_per_subject_per_roi_per_param[params] = zscore_per_subject_per_roi_per_param[params].applymap(
            StatisticsWrapper.computed_zscore_per_values)

        return zscore_per_subject_per_roi_per_param

    @staticmethod
    def calc_z_score_on_mean_per_subject(subjects_raw_data, params):
        """
        First calculates the mean of each ROI
        Than do z_score on each parameter per all ROIs (depends on the number of ROIs)
        :param subjects_raw_data: given data
        :param params: given relevant params to analyze.
        :return:
        """
        mean_per_subject_per_roi_per_param = StatisticsWrapper.calc_mean_per_subject_per_parameter_per_ROI(
            subjects_raw_data, params)
        # mean_per_subject_per_roi_per_param[PARAMETERS] = mean_per_subject_per_roi_per_param[PARAMETERS].applymap(lambda x: stats.zscore(x.groupby('subjects')[PARAMETERS]))
        # mean_per_subject_per_roi_per_param[PARAMETERS] = mean_per_subject_per_roi_per_param[PARAMETERS].apply(lambda x: stats.zscore(x.groupby('subjects'))[PARAMETERS])
        # mean_per_subject_per_roi_per_param[PARAMETERS] = mean_per_subject_per_roi_per_param[PARAMETERS].apply(lambda x: stats.zscore(x)) #TODO: SUPER IMPORTNAT NOT GOOD! NEED TO BE FIXED SINCE IT IS DOING ZSCORE ON ALL DATA PER PARAm AND NOT PER SUBJECT!!
        mean_per_subject_per_roi_per_param[params] = mean_per_subject_per_roi_per_param[params + ['subjects']].groupby(
            "subjects").apply(stats.zscore)
        # TODO apply zscore function on all means of all keys!
        return mean_per_subject_per_roi_per_param

    @staticmethod
    def calc_z_score_on_mean_per_subject2(subjects_raw_data, params):
        """
        First do z_score on each cell, than takes the everage of each z_score - doesn't depend
        on num of ROIS
        :param subjects_raw_data: given data
        :param params: given relevant params to analyze.
        :return:
        """
        mean_per_subject_per_roi_per_param = StatisticsWrapper.calc_z_score_per_subject(subjects_raw_data, params)
        # mean_per_subject_per_roi_per_param[PARAMETERS] = mean_per_subject_per_roi_per_param[PARAMETERS].applymap(lambda x: stats.zscore(x.groupby('subjects')[PARAMETERS]))
        # mean_per_subject_per_roi_per_param[PARAMETERS] = mean_per_subject_per_roi_per_param[PARAMETERS].apply(lambda x: stats.zscore(x.groupby('subjects'))[PARAMETERS])
        # mean_per_subject_per_roi_per_param[PARAMETERS] = mean_per_subject_per_roi_per_param[PARAMETERS].apply(lambda x: stats.zscore(x)) #TODO: SUPER IMPORTNAT NOT GOOD! NEED TO BE FIXED SINCE IT IS DOING ZSCORE ON ALL DATA PER PARAm AND NOT PER SUBJECT!!
        return StatisticsWrapper.calc_mean_per_subject_per_parameter_per_ROI(mean_per_subject_per_roi_per_param, params)
        # TODO apply zscore function on all means of all keys!

    @staticmethod
    def robust_scaling(cell):
        """
        Calculates the robust scaling
        :param cell: given cell
        :return: return the value after it was normilzed with robust-scaling
        """
        return list((np.array(cell) - np.median(cell)) / (np.quantile(cell, 0.75) - np.quantile(cell, 0.25)))

    @staticmethod
    def robust_scaling_mean(cell):
        """
        Mean on the robust scaling values.
        :param cell: given cell.
        :return: The mean of the robust-scaled cell.
        """
        return np.mean(StatisticsWrapper.robust_scaling(cell))

    @staticmethod
    def calc_mean_robust_scaling_per_subject_per_parameter_per_ROI(subjects_raw_data, params):
        """
        Calculated mean per subject per parameter per ROI after normalizing it by robust scaling
        :param subjects_raw_data: given data
        :param params: given relevant params to analyze.
        :return: the mean of the robust-scaled data.
        """
        # scaler = RobustScaler()
        robust_scaling_per_subject_per_roi_per_param = subjects_raw_data.copy()
        # robust_scaling_per_subject_per_roi_per_param = pd.DataFrame(scaler.fit_transform(robust_scaling_per_subject_per_roi_per_param), columns=PARAMETERS)
        robust_scaling_per_subject_per_roi_per_param[params] = robust_scaling_per_subject_per_roi_per_param[
            params].applymap(StatisticsWrapper.robust_scaling_mean)
        return robust_scaling_per_subject_per_roi_per_param

    @staticmethod
    def chose_relevant_data(data: pd.DataFrame, rois_to_analyze=SUB_CORTEX_DICT,
                            params_to_work_with: List[str] = None) -> pd.DataFrame:
        """
        Chose only the data with relevant ROIs
        :param data: given data
        :param rois_to_analyze: rois to analyze in the data
        :param params_to_work_with: parameter to work with
        :return: data only with given rois and params
        """
        if params_to_work_with != None:
            cols = data.columns
            params_to_remove = set(PARAMETERS_W_D_TV_R1_AND_R2S) - set(params_to_work_with)
            if all(param in cols for param in params_to_remove):
                return data[data["ROI"].isin(list(rois_to_analyze.keys()))].drop(params_to_remove, axis=1)
            else:
                print("Didn't drop params since not all in the data")

        return data[data["ROI"].isin(list(rois_to_analyze.keys()))]

    @staticmethod
    def seperate_data_to_two_groups(data: pd.DataFrame, separating_column: str, threshold: Any) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Seperating data into two groups according to
        :param data: given data
        :param separating_column: seperating columns, like 'Age', 'Sex'
        :param threshold:
        :return:
        """
        if type(threshold) == int or type(threshold) == float:
            group_a = data[data[separating_column] <= threshold]
            group_b = data[data[separating_column] > threshold]
        else:
            group_a = data[data[separating_column] == threshold]
            group_b = data[data[separating_column] != threshold]
        return group_a, group_b

    @staticmethod
    def t_test_per_parameter_per_area(data1, data2, wanted_dict, compare_column, data1_name, data2_name):
        """
        Calculates T-Test for each parameter per area between data1 and data2 and print them
        :param data1: df to compare with data2
        :param data2: df to compare with data1
        :param wanted_dict: only wanted
        :param compare_column: the column to get the data from and compare it
        :param data1_name:
        :param data2_name:
        :return: None
        """
        for col_name in data1.columns:
            if col_name == 'subjects' or col_name == 'ROI' or col_name == 'Age' \
                    or col_name == "Gender" or col_name == "ROI_name":
                continue
            for area in wanted_dict.keys():
                results = stats.ttest_ind(a=data1[col_name][data1[compare_column] == area].to_numpy(),
                                          b=data2[col_name][data2[compare_column] == area].to_numpy())
                if results.pvalue <= 0.05:
                    print(f"T_Test for {col_name} between {data1_name} and {data2_name} in {wanted_dict[area]}",
                          results)

    @staticmethod
    def plot_values_of_two_groups_per_roi(ROIs, info_per_ROI_per_param1: List[int], info_per_ROI_per_param2: List[int],
                                          param, name_group_a, name_group_b, save_address: str, info_name: str):
        """
        Plot values of two groups (for example given SD/Means of all values per ROI)
        :param ROIs: given ROIs to check
        :param info_per_ROI_per_param1:  info (SD/Means/etc') of each ROI per parameter
        :param info_per_ROI_per_param2: info (SD/Means/etc') of each ROI per parameter
        :param param: given param which we check (r1,r2s,diffusion etc')
        :param name_group_a: name of group a (young, male etc')
        :param name_group_b: name of group b
        :param save_address: the address to save the output to.
        :param info_name: Standard Deviation / Means/ Etc
        :return:
        """
        plt.plot([str(i) for i in ROIs], info_per_ROI_per_param1, mfc="orange", mec='k', ms=7, marker="o",
                 linestyle="None")
        plt.plot([str(i) for i in ROIs], info_per_ROI_per_param2, mfc="blue", mec='k', ms=7, marker="o",
                 linestyle="None")
        plt.title(f"{info_name} of {param} per ROI")
        plt.legend([f'{name_group_a}', f'{name_group_b}'])
        plt.ylabel(f"{info_name}")
        plt.xlabel("ROIs")

        if not os.path.exists(save_address + f"/{info_name} /"):
            os.makedirs(save_address + f"/{info_name} /")

        plt.savefig(save_address + f"/{info_name} /" + f"{param}_distribution" + '.png')
        plt.show()

    @staticmethod
    def computed_std_per_parameter(data1, data2, parameters, ROIS, name_group_a, name_group_b, save_address,
                                   visualize=False, log=False, project_name=None):
        """
        Computes SD per parameter per ROI for young and adults.
        :param data1: group a data
        :param data2: group b data
        :param parameters: given parameters to work on the data
        :param ROIS: rois
        :param name_group_a:
        :param name_group_b:
        :param save_address: the save address (path) to where the output will be saved.
        :param visualize: true - create a grpah to visualize the data, false otherwise.
        :return:
        """
        for param in parameters:
            wandb_run = wandb.init(
                project=project_name,
                name=f'{param}_std'
            )

            std_per_ROI_per_param1 = []
            std_per_ROI_per_param2 = []

            for ROI in ROIS:
                std1 = data1[param][data1['ROI'] == ROI].std()
                std2 = data2[param][data2['ROI'] == ROI].std()
                std_per_ROI_per_param1.append(std1)
                std_per_ROI_per_param2.append(std2)

            if visualize:
                StatisticsWrapper.plot_values_of_two_groups_per_roi(ROIS, std_per_ROI_per_param1,
                                                                    std_per_ROI_per_param2, param, name_group_a,
                                                                    name_group_b, save_address,
                                                                    "Standard Deviation")

            if log:
                wandb_run.log({f'{param}_std': wandb.plot.line_series(
                    xs=ROIS,
                    ys=[std_per_ROI_per_param1, std_per_ROI_per_param2],
                    keys=[name_group_a, name_group_b],
                    title=f'{param} Rois Standard Deviation',
                    xname="ROI",
                )})

                wandb_run.finish()


    @staticmethod
    def plot_data_per_param_per_roi_next_to_each_other(data1, data2, name_group_a, name_group_b, save_address):
        """
        Plot data per parameter per roi next to each other - group a near group b
        :param data1: data of group a
        :param data2: data of group b
        :param name_group_a: name of group a (old, etc')
        :param name_group_b: name of group b (young, etc')
        :param save_address: save address
        :return: None
        """
        data = pd.concat([data2, data1])
        data = data.assign(Mature=np.where(data['Age'] >= AGE_THRESHOLD, name_group_b, name_group_a))
        for col_name in data.columns:
            if col_name == 'subjects' or col_name == 'ROI' or col_name == 'Age' \
                    or col_name == "Gender" or col_name == "ROI_name" or col_name == 'Mature':
                continue
            range_y_values = [min(data[col_name]) - min(data[col_name]) / 100,
                              max(data[col_name]) + max(data[col_name]) / 100]
            sns.boxplot(x="ROI", y=col_name, data=data, showmeans=True, hue='Mature',
                        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black",
                                   "markersize": "3"})
            plt.ylim(range_y_values)
            plt.ylabel(col_name)
            plt.xlabel("ROI")
            plt.suptitle(f"{col_name} per ROI for all subjects")

            if not os.path.exists(save_address + "/distribution/"):
                os.makedirs(save_address + "/distribution/")
            plt.savefig(save_address + "/distribution/" + f"{col_name}_distribution" + '.png')
            plt.show()

            StatisticsWrapper.plot_data_per_parameter_for_rois(data1, data2, "", YOUNG, OLD)
        return

    @staticmethod
    def plot_data_per_parameter_for_rois(data1, data2, description_data, compare_val1, compare_val2, wanted_dict=None):
        """
         Plot data per parameter per roi next to each other - group a near group b, but each group in different
         graph inside the graphs
        :param data1: data of group a
        :param data2: data of group b
        :param description_data: description data
        :param compare_val1: name val 1
        :param compare_val2: name val 2
        :param wanted_dict: wanted dictionary
        :return:
        """
        for col_name in data1.columns:
            if col_name == 'subjects' or col_name == 'ROI' or col_name == 'Age' \
                    or col_name == "Gender" or col_name == "ROI_name":
                continue
            range_y_values = [
                min(min(data1[col_name]), min(data2[col_name])) - min(min(data1[col_name]), min(data2[col_name])) / 100,
                max(max(data1[col_name]), max(data2[col_name])) + max(max(data1[col_name]), max(data2[col_name])) / 100]
            # todo: Try to box each ROI next to each other with different colors
            fig, axs = plt.subplots(1, 2)
            sns.boxplot(x="ROI", y=col_name, data=data1, showmeans=True,
                        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black",
                                   "markersize": "3"}, ax=axs[0])
            axs[0].set_title("YOUNG")
            plt.ylim(range_y_values)
            sns.boxplot(x="ROI", y=col_name, data=data2, showmeans=True,
                        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black",
                                   "markersize": "3"}, ax=axs[1])
            axs[1].set_title("OLD")
            plt.suptitle(f"{col_name} per ROI for all subjects {description_data}")
            plt.ylim(range_y_values)
            plt.tight_layout()
            plt.ylabel(col_name)
            plt.xlabel("ROI")
            plt.show()

    @staticmethod
    def calculate_pvalues(df):
        """
        Calculated p value for each
        :param df:
        :return:
        """
        df = df.dropna()._get_numeric_data()
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
        return pvalues

    @staticmethod
    def plot_hierarchical_correlation_map(sub_name, df_corr, colums_name, save_address=None, file_name=None):
        """
        Plot Hierarchical Correlation Map in a subject per ROI, with the given correlation matrix.
        :param sub_name: subject name
        :param df_corr: correlation matrix between ROIS, that does the correlation on vector of given params
        :param colums_name: Names of ROIs
        :param save_address: The path to where the file will be saved
        :param file_name: The name the file will be saved
        :return:
        """
        plt.figure(figsize=(250, 250))
        sns.set(font_scale=0.75)
        plt.suptitle(str(sub_name) + " Correlation Map\n")
        df_corr.columns = colums_name
        df_corr.index = colums_name
        sns.clustermap(df_corr, cmap='coolwarm',
                       # xticklabels=colums_name,
                       # yticklabels=colums_name,
                       annot=True)
        if not (save_address is None or file_name is None):
            if not os.path.exists(save_address):
                os.makedirs(save_address)
            plt.savefig(save_address + file_name + sub_name + '.png')
        # plt.show()

    @staticmethod
    def calculate_correlation_per_data(df, params_to_work_with, ROIs_to_analyze, group_name, save_address):
        """
        Calculates correlation in each subject, between all ROI's when each ROI has a vector of parameters.
        :param df: df containing all subject and their values in each ROI and in each Parameter.
        :param params_to_work_with: the parameters to compute the correlations with.
        :param ROIs_to_analyze: ROIs to analyze
        :param group_name: the df's group name (all, adult, young, etc)
        :param save_address: The save address
        :return: None
        """
        relevant_ROIs = list(ROIs_to_analyze.values())
        all_correlations = np.zeros((len(relevant_ROIs),
                                     len(relevant_ROIs)))  # Variable which will have sum of all correlations and then divide it with num of subject
        for subject_name in df.subjects.unique():
            # Compute correlation only with the current subject between all rois with given parameters.
            df_corr = df[df['subjects'] == subject_name][params_to_work_with].T.corr()
            StatisticsWrapper.plot_hierarchical_correlation_map(subject_name, df_corr, relevant_ROIs, save_address,
                                                                HIERARCHICAL_CLUSTERING_FILE)
            all_correlations += df_corr.to_numpy()

            a = StatisticsWrapper.calculate_pvalues(df[df['subjects'] == subject_name][params_to_work_with].T)
            if a[a < 0.05].count().sum() > 6:
                print(StatisticsWrapper.calculate_pvalues(df[df['subjects'] == subject_name][params_to_work_with].T))
        all_correlations /= len(df.subjects.unique())
        all_correlations = pd.DataFrame(all_correlations)
        StatisticsWrapper.plot_hierarchical_correlation_map(f"Mean Of {group_name} Subjects", all_correlations,
                                                            relevant_ROIs,
                                                            save_address, HIERARCHICAL_CLUSTERING_FILE)

    @staticmethod
    def plot_values_per_parameter_per_roi(data, params, rois, save_address):
        """
        This function gets the data frame's data, parametersm rois and shows for each roi all the values
        per parameter ->
        each graph is per ROI, and shows a scatter plot for each parameter. This may help understand the
        correlation and see which parameter is more siginificant in the correlation!
        :param data: given data frame on it we check.
        :param params: given params to work with
        :param rois: relvant ROIS
        :param save_address: save address for graphs
        :return:
        """
        save_address_for_func = os.path.join(save_address, "Visual_Corr")
        if save_address:
            if not os.path.exists(save_address_for_func):
                os.makedirs(save_address_for_func)
        colors = ['blue', 'red', 'yellow', 'green', 'black', 'gray', 'pink', 'silver', 'orange', 'gold']
        for subject in np.unique(data['subjects']):
            cur_data = data[data['subjects'] == subject]
            # Enough to check on certain subject but if want to check on all - you can change it
            if subject == "H018_AS":
                # plt.bar(ROIS,std_per_ROI_per_param)
                for i in range(len(rois)):
                    for j in range(i + 1, len(rois)):
                        param_info_per_roi_per_subject = cur_data[cur_data['ROI'] == rois[i]][params]
                        param_info_per_roi_per_subject2 = cur_data[cur_data['ROI'] == rois[j]][params]
                        all_scatter_plots = []
                        for k in range(len(params)):
                            all_scatter_plots.append(plt.scatter(list(param_info_per_roi_per_subject.iloc[0])[k],
                                                                 list(param_info_per_roi_per_subject2.iloc[0])[k],
                                                                 color=colors[k]))
                        plt.legend(all_scatter_plots, params, scatterpoints=1, ncol=3, fontsize=8)
                        plt.title(f"{subject}\n info of {params} per {SUB_CORTEX_DICT[rois[i]]} \n and "
                                  f"{SUB_CORTEX_DICT[rois[j]]}")
                        plt.ylabel(SUB_CORTEX_DICT[rois[j]])
                        plt.xlabel(SUB_CORTEX_DICT[rois[i]])
                        if save_address:
                            plt.savefig(save_address_for_func + "/" +
                                        f"cor_{SUB_CORTEX_DICT[rois[i]]}_and_{SUB_CORTEX_DICT[rois[j]]}.png")
                        plt.show()
