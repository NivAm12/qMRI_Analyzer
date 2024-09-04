import os
import enum
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f
from typing import List, Any, Tuple
import wandb
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
import constants
from .plots import PlotsManager
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
import math


# -------------------- Enums for statistical actions -------------------- #
class Actions(enum.Enum):
    z_score = 1  # Z Score on data - per subject, per parameter, per ROI
    z_score_means = 2  # Z Score on means of all subjects, per parameters, per its ROI
    means_per_subject = 3  # Means on each Subject, per its ROI (per parameter)
    robust_scaling = 4  # subtracting the median and dividing by the interquartile range


# -------------- Dictionaries of raw_data_type to their input path and output Dir ------------- #
RAW_DATA_NORMALIZER_PATH = {constants.RAW_DATA: constants.PATH_TO_RAW_DATA,
                            constants.Z_SCORE: constants.PATH_TO_RAW_DATA_Z_SCORED,
                            constants.ROBUST_SCALING: constants.PATH_TO_RAW_DATA_ROBUST_SCALED}

RAW_DATA_NORMALIZER_OUTPUT_DIR = {constants.RAW_DATA: constants.RAW_DATA_DIR,
                                  constants.Z_SCORE: constants.RAW_DATA_Z_SCORED_DIR,
                                  constants.ROBUST_SCALING: constants.RAW_DATA_ROBUST_SCALED_DIR}

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
        std_per_subject_per_roi_per_param[params] = std_per_subject_per_roi_per_param[params].applymap(
            np.std)
        return std_per_subject_per_roi_per_param

    @staticmethod
    def calc_median_per_subject_per_parameter_per_ROI(subjects_raw_data, params):
        """
        calculates the median per subject per parameter per ROI.
        :param subjects_raw_data: given data
        :param params: given relevant params to analyze.
        :return: the data after being manipulated to means
        """
        mean_per_subject_per_roi_per_param = subjects_raw_data.copy()
        mean_per_subject_per_roi_per_param[params] = mean_per_subject_per_roi_per_param[params].applymap(
            np.median)
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

        mean_per_subject_per_roi_per_param[params] = mean_per_subject_per_roi_per_param[params + ['subjects']].groupby(
            "subjects").apply(stats.zscore)

        # TODO apply zscore function on all means of all keys!
        return mean_per_subject_per_roi_per_param

    @staticmethod
    def calc_z_score_on_mean_per_subject2(subjects_raw_data, params):
        """
        First do z_score on each cell, then takes the average of each z_score - doesn't depend
        on num of ROIS
        :param subjects_raw_data: given data
        :param params: given relevant params to analyze.
        :return:
        """
        mean_per_subject_per_roi_per_param = StatisticsWrapper.calc_z_score_per_subject(
            subjects_raw_data, params)

        return StatisticsWrapper.calc_mean_per_subject_per_parameter_per_ROI(mean_per_subject_per_roi_per_param, params)
        # TODO apply zscore function on all means of all keys!

    @staticmethod
    def robust_scaling(cell):
        """
        Calculates the robust scaling
        :param cell: given cell
        :return: return the value after it was normalized with robust-scaling
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
    def chose_relevant_data(data: pd.DataFrame, rois_to_analyze=constants.SUB_CORTEX_DICT, raw_params: List[str] = None,
                            params_to_work_with: List[str] = None) -> pd.DataFrame:
        """
        Chose only the data with relevant ROIs
        :param data: given data
        :param rois_to_analyze: rois to analyze in the data
        :param params_to_work_with: parameter to work with
        :param raw_params: original params of the data
        :return: data only with given rois and params
        """
        if params_to_work_with is not None:
            cols = data.columns
            params_to_remove = set(raw_params) - set(params_to_work_with)
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
    def t_test_per_parameter_per_area(data1, data2, rois, compare_column, params, title):
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
        t_test_params = {}

        for param in params:
            num_of_significance = 0
            for area in rois.keys():
                results = stats.ttest_ind(a=data1[param][data1[compare_column] == area].to_numpy(),
                                          b=data2[param][data2[compare_column] == area].to_numpy())
                significance = results.pvalue <= 0.05
                # print(f"T_Test for {param} {rois[area]} significance:{significance}, results: {results}")

                if significance:
                    num_of_significance += 1

            t_test_params[param] = (num_of_significance / len(rois.keys())) * 100

        plt.figure(figsize=(10, 4))
        plt.title(title)
        plt.bar(t_test_params.keys(), t_test_params.values())
        plt.xlabel('Param')
        plt.ylabel('ROIs % with significance difference')

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

        # if not os.path.exists(save_address + f"/{info_name} /"):
        #     os.makedirs(save_address + f"/{info_name} /")

        # plt.savefig(save_address + f"/{info_name} /" + f"{param}_distribution" + '.png')
        plt.show()

    @staticmethod
    def computed_std_per_parameter(data1, data2, parameters, ROIS, name_group_a, name_group_b, save_address=None,
                                   visualize=False, project_name=None):
        """
        Computes SD per parameter per ROI for young and adults.
        :param data1: group a data
        :param data2: group b data
        :param parameters: given parameters to work on the data
        :param ROIS: rois
        :param name_group_a:
        :param name_group_b:
        :param save_address: the save address (path) to where the output will be saved.
        :param visualize: true - create a graph to visualize the data, false otherwise.
        :param project_name: wandb project name
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

            if project_name:
                wandb_run.log({f'{param}_std': wandb.plot.line_series(
                    xs=ROIS,
                    ys=[std_per_ROI_per_param1, std_per_ROI_per_param2],
                    keys=[name_group_a, name_group_b],
                    title=f'{param} Rois Standard Deviation',
                    xname="ROI",
                )})

                wandb_run.finish()

    @staticmethod
    def plot_data_per_param_per_roi_next_to_each_other(data1, data2, params, name_group_a, name_group_b,
                                                       save_address=None,
                                                       project_name=None):
        """
        Plot data per parameter per roi next to each other - group a near group b
        :param data1: data of group a
        :param data2: data of group b
        :param name_group_a: name of group a (old, etc')
        :param name_group_b: name of group b (young, etc')
        :param save_address: save address
        :param project_name: wandb project name
        :return: None
        """
        data = pd.concat([data2, data1])
        data = data.assign(Mature=np.where(
            data['Age'] >= constants.AGE_THRESHOLD, name_group_b, name_group_a))

        for param in params:
            if "Slope" in param:
                continue

            plt.figure(figsize=(18, 8))
            sns.boxplot(x="ROI", y=param, data=data, showmeans=True, hue='Mature', width=0.5,
                        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black",
                                   "markersize": "3"}).set_title(param)

            if project_name:
                wandb_run = wandb.init(
                    project=project_name,
                    name=f'{param}boxplot'
                )

                wandb_run.log({f'{param}': wandb.Image(plt)})
                wandb_run.finish()
                plt.close()

            # plt.close()
            # StatisticsWrapper.plot_data_per_parameter_for_rois(data1, data2, "", name_group_a, name_group_b)

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
                min(min(data1[col_name]), min(data2[col_name])) -
                min(min(data1[col_name]), min(data2[col_name])) / 100,
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
            plt.suptitle(
                f"{col_name} per ROI for all subjects {description_data}")
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
        df_cols = pd.DataFrame(columns=df.columns)
        p_values = df_cols.transpose().join(df_cols, how='outer')

        for r in df.columns:
            for c in df.columns:
                p_values[r][c] = round(pearsonr(df[r], df[c])[1], 4)

        return p_values

    @staticmethod
    def plot_hierarchical_correlation_map(sub_name, corr_mat, colunm_name):
        """
        Plot Hierarchical Correlation Map in a subject per ROI, with the given correlation matrix.
        :param sub_name: subject name
        :param corr_mat: correlation matrix between ROIS, that does the correlation on vector of given params
        :param colunm_name: Names of ROIs
        :return:
        """
        df_corr = pd.DataFrame(corr_mat)

        cluster_map = sns.clustermap(df_corr,
                                     cmap='coolwarm',
                                     row_cluster=False,
                                     fmt=".2f",
                                     figsize=(20, 10))

        cluster_map.fig.suptitle(str(sub_name) + " Correlation Map\n")
        plt.show()

    @staticmethod
    def calculate_correlation_per_data(df, params_to_work_with, ROIs_to_analyze, group_name, save_address,
                                       project_name=None):
        """
        Calculates correlation in each subject, between all ROI's when each ROI has a vector of parameters.
        :param df: df containing all subject and their values in each ROI and in each Parameter.
        :param params_to_work_with: the parameters to compute the correlations with.
        :param ROIs_to_analyze: ROIs to analyze
        :param group_name: the df's group name (all, adult, young, etc)
        :param save_address: The save address
        :return: None
        """
        relevant_rois = list(df.ROI.unique())
        all_correlations = np.zeros((len(relevant_rois),
                                     len(relevant_rois)))

        for subject_name in df.subjects.unique():
            # Compute correlation only with the current subject between all rois with given parameters.
            df_corr = df[df['subjects'] ==
                         subject_name][params_to_work_with].T.corr()
            all_correlations += df_corr.to_numpy()

        all_correlations /= len(df.subjects.unique())

        StatisticsWrapper.plot_hierarchical_correlation_map(f"Mean Of {group_name} Subjects", all_correlations,
                                                            relevant_rois)

        if project_name:
            wandb_run = wandb.init(
                project=project_name,
                name=f'{group_name} hierarchical correlation'
            )

            wandb_run.log({f'{group_name}': wandb.Image(plt)})
            wandb_run.finish()
            plt.close()

    @staticmethod
    def plot_values_per_parameter_per_roi(data, params, rois, save_address=None):
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
        if save_address:
            save_address_for_func = os.path.join(save_address, "Visual_Corr")
            if not os.path.exists(save_address_for_func):
                os.makedirs(save_address_for_func)
        colors = ['blue', 'red', 'yellow', 'green', 'black',
                  'gray', 'pink', 'silver', 'orange', 'gold']
        for subject in np.unique(data['subjects']):
            cur_data = data[data['subjects'] == subject]
            # Enough to check on certain subject but if want to check on all - you can change it
            if subject == "H018_AS":
                # plt.bar(ROIS,std_per_ROI_per_param)
                for i in range(len(rois)):
                    for j in range(i + 1, len(rois)):
                        param_info_per_roi_per_subject = cur_data[cur_data['ROI']
                                                                  == rois[i]][params]
                        param_info_per_roi_per_subject2 = cur_data[cur_data['ROI']
                                                                   == rois[j]][params]
                        all_scatter_plots = []
                        for k in range(len(params)):
                            all_scatter_plots.append(plt.scatter(list(param_info_per_roi_per_subject.iloc[0])[k],
                                                                 list(param_info_per_roi_per_subject2.iloc[0])[
                                k],
                                color=colors[k]))
                        # plt.legend(all_scatter_plots, params, scatterpoints=1, ncol=3, fontsize=8)
                        # plt.title(f"{subject}\n info of {params} per {constants.SUB_CORTEX_DICT[rois[i]]} \n and "
                        #           f"{constants.SUB_CORTEX_DICT[rois[j]]}")
                        # plt.ylabel(constants.SUB_CORTEX_DICT[rois[j]])
                        # plt.xlabel(constants.SUB_CORTEX_DICT[rois[i]])
                        # if save_address:
                        #     plt.savefig(save_address_for_func + "/" +
                        #                 f"cor_{constants.SUB_CORTEX_DICT[rois[i]]}_and_{constants.SUB_CORTEX_DICT[rois[j]]}.png")
                        # plt.show()

    @staticmethod
    def hierarchical_clustering(data: pd.DataFrame, params_to_work_with: list, linkage_metric: str,
                                project_name: str = None, title: str = None, show: bool = True):
        subjects = data.groupby('subjects')
        relevant_rois = list(data.ROI_name.unique())
        distances = np.zeros((len(relevant_rois),
                              len(relevant_rois)))

        for subject_name, subject_df in subjects:
            df = subject_df[params_to_work_with]
            dist = pdist(df, metric='cosine')
            distance_matrix = pd.DataFrame(squareform(
                dist), index=relevant_rois, columns=relevant_rois)
            distances += distance_matrix.to_numpy()

        distances /= data.subjects.nunique()
        clusters = linkage(distances, method=linkage_metric)

        # flat_clusters = fcluster(clusters, t=2.5, criterion='distance')


        dendrogram_data = PlotsManager.create_and_plot_dendrogram(clusters,
                                                                  relevant_rois,
                                                                  title, linkage_metric, project_name, show=show)

        clustered_indexes = {}
        for cluster_label in np.unique(dendrogram_data['color_list']):
            color_mask = np.array(dendrogram_data['leaves_color_list']) == cluster_label
            clustered_indexes[cluster_label] = np.array(dendrogram_data['ivl'])[color_mask]

        return {'clusters': clusters, 'dendrogram_data': dendrogram_data, 'clusters_groups': clustered_indexes}

    @staticmethod
    def roi_correlations(data: pd.DataFrame, params_to_work_with: list, rois: list,
                         group_title: str = None, project_name: str = None, method="pearson", show: bool = True):
        subjects = data.groupby('subjects')
        relevant_rois = list(data.ROI_name.unique())
        correlations = np.zeros((len(relevant_rois),
                                 len(relevant_rois)))

        for subject_name, subject_df in subjects:
            df_corr = subject_df[params_to_work_with].T.corr(method=method)
            correlations += df_corr.to_numpy()

        correlations /= data.subjects.nunique()
        # labels = [label[4:] for label in relevant_rois]  # remove prefix as 'ctx'
        correlations_df = pd.DataFrame(
            correlations, index=relevant_rois, columns=relevant_rois)

        # reorder the dataframe to match the clustering order
        correlations_df = correlations_df.reindex(rois)
        correlations_df = correlations_df[rois]

        # plot the heatmap
        if show:
            PlotsManager.plot_heatmap(correlations_df, group_title, project_name)

        return correlations_df

    @staticmethod
    def roi_correlations_std(data: pd.DataFrame, params_to_work_with: list, rois: list,
                             title: str = None, project_name: str = None, method="pearson"):
        subjects = data.groupby('subjects')
        relevant_rois = list(data.ROI_name.unique())
        correlations_matrices = []

        for subject_name, subject_df in subjects:
            df_corr = subject_df[params_to_work_with].T.corr(method=method)
            correlations_matrices.append(df_corr.to_numpy())

        # Combine matrices into a single array along a new axis
        combined_matrix = np.stack(correlations_matrices, axis=0)

        # Calculate standard deviation along the first axis (across all matrices)
        std_matrix = np.std(combined_matrix, axis=0)

        # labels = [label[4:] for label in relevant_rois]  # remove prefix as 'ctx'
        correlations_df = pd.DataFrame(
            std_matrix, index=relevant_rois, columns=relevant_rois)

        # reorder the dataframe to match the clustering order
        correlations_df = correlations_df.reindex(rois)
        correlations_df = correlations_df[rois]

        # plot the heatmap
        PlotsManager.plot_heatmap(correlations_df, title, project_name)

        return correlations_df

    @staticmethod
    def roi_distances(data: pd.DataFrame, params_to_work_with: list, rois: list = None,
                      method = None, title: str = None, project_name: str = None, show: bool = True):
        subjects = data.groupby('subjects')
        relevant_rois = list(data.ROI_name.unique())
        distance_matrices = []

        for subject_name, subject_df in subjects:
            dist_matrix = method(
                subject_df[params_to_work_with].values, subject_df[params_to_work_with].values)
            distance_matrices.append(dist_matrix)

        # Calculate mean distance along the first axis (across all matrices)
        mean_distance_matrix = np.mean(distance_matrices, axis=0)

        # labels = [label[4:] for label in relevant_rois]  # remove prefix as 'ctx'
        distance_df = pd.DataFrame(
            mean_distance_matrix, index=relevant_rois, columns=relevant_rois)

        if rois:
            # reorder the dataframe to match the clustering order
            distance_df = distance_df.reindex(rois)
            distance_df = distance_df[rois]

        if show:
            # plot the heatmap
            PlotsManager.plot_heatmap(distance_df, title, project_name)

        return distance_df

    @staticmethod
    def roi_distances_by_age(data: pd.DataFrame, params_to_work_with: list,
                             project_name: str = None):
        subjects = data.groupby('subjects')
        relevant_rois = list(data.ROI_name.unique())
        labels = [label[4:]
                  for label in relevant_rois]  # remove prefix as 'ctx'

        num_rois = len(labels)
        fig, axes = plt.subplots(num_rois, 1, figsize=(10, 6*num_rois))

        for roi_index, roi in enumerate(labels):
            age_values = []
            correlation_values = []
            for subject_name, subject_df in subjects:
                dist_matrix = pd.DataFrame(cosine_similarity(
                    subject_df[params_to_work_with].values, subject_df[params_to_work_with].values))
                dist_matrix.index = labels
                dist_matrix.columns = labels
                dist_matrix['dist_mean'] = dist_matrix.apply(np.mean, axis=1)
                age_values.append(subject_df.Age.iloc[0])
                correlation_values.append(dist_matrix.loc[roi]['dist_mean'])

            ax = axes[roi_index]
            ax.scatter(age_values, correlation_values)

            # Perform linear regression to get slope (m) and intercept (b)
            # degree 1 for linear regression
            m, b = np.polyfit(age_values, correlation_values, 1)

            # Generate x values for the regression line
            x_line = np.linspace(min(age_values), max(age_values), 100)

            # Calculate y values for the regression line using the equation y = mx + b
            y_line = m * x_line + b

            # Plot the regression line on the scatter plot
            ax.plot(x_line, y_line, color='red')  # adjust color as desired

            ax.set_title(f'ROI: {roi}')
            ax.set_xlabel('Age')
            ax.set_ylabel('Mean Distances')

        plt.show()

    @staticmethod
    def roi_correlations_by_age_by_each_roi(data: pd.DataFrame, params_to_work_with: list,
                                            title: str = None, project_name: str = None, method="pearson"):
        subjects = data.groupby('subjects')
        relevant_rois = list(data.ROI_name.unique())
        labels = [label[4:]
                  for label in relevant_rois]  # remove prefix as 'ctx'

        num_rois = len(labels)
        fig, axes = plt.subplots(num_rois, 1, figsize=(10, 6*num_rois))

        for roi_index, roi in enumerate(labels):
            age_values = []
            correlation_values = []
            for subject_name, subject_df in subjects:
                df_corr = subject_df[params_to_work_with].T.corr(method=method)
                df_corr.index = labels
                df_corr.columns = labels
                df_corr['corr_mean'] = df_corr.apply(np.mean, axis=1)
                age_values.append(subject_df.Age.iloc[0])
                correlation_values.append(df_corr.loc[roi]['corr_mean'])

            ax = axes[roi_index]
            ax.scatter(age_values, correlation_values)

            # Perform linear regression to get slope (m) and intercept (b)
            # degree 1 for linear regression
            m, b = np.polyfit(age_values, correlation_values, 1)

            # Generate x values for the regression line
            x_line = np.linspace(min(age_values), max(age_values), 100)

            # Calculate y values for the regression line using the equation y = mx + b
            y_line = m * x_line + b

            # Plot the regression line on the scatter plot
            ax.plot(x_line, y_line, color='red')  # adjust color as desired

            ax.set_title(f'ROI: {roi}')
            ax.set_xlabel('Age')
            ax.set_ylabel('Mean Correlation')

        plt.show()

    @staticmethod
    def calculate_cv_for_subjects(data_groups, group_by_param, params, x_axis, use_reg=False, fig_size=(20, 8), connect_scatter=False):
        for param in params:
            plt.figure(figsize=fig_size)
            for data, color, label in data_groups:
                # Calculate CV params
                means = data.groupby(group_by_param)[[param, x_axis]].mean()
                stds = data.groupby(group_by_param)[param].std()
                cv = (stds / means[param])

                if use_reg:
                    model = LinearRegression()
                    x = np.array(means[x_axis]).reshape(-1, 1)
                    y = np.array(cv).reshape(-1, 1)
                    model.fit(x, y)
                    # Get the slope and intercept
                    slope = model.coef_[0]
                    intercept = model.intercept_
                    x_fit = np.linspace(
                        min(means[x_axis]), max(means[x_axis]), 100)
                    x_axis_to_use = means[x_axis]
                    # Get R-squared score
                    r2 = r2_score(y, model.predict(x))
                else:
                    x_axis_to_use = [str(int(i)) for i in means[x_axis]]

                cv_data = pd.DataFrame({
                    'CV': cv,
                    x_axis: x_axis_to_use
                })

                # Create the plot
                plt.scatter(cv_data[x_axis], cv_data['CV'],
                            color=color, label=label, s=50, alpha=0.7)
                if connect_scatter:
                    plt.plot(cv_data[x_axis], cv_data['CV'])
                if use_reg:
                    plt.annotate(
                        f'R2: {r2}', (min(means[x_axis]) * 0.9, max(cv_data['CV'])), fontsize=10)
                    plt.plot(x_fit, slope * x_fit + intercept, color='red')
                plt.xlabel(x_axis)
                plt.ylabel('CV')

            plt.title(f'{param}')
            plt.grid(True)
            plt.legend()

    @staticmethod
    def calculate_mean_std_for_rois(data_groups, rois, params, fig_size=(20, 8), t_test_params=None):
        rois_labels = [str(roi) for roi in rois]
        groups_rois_std = {}

        plt.figure(figsize=fig_size)
        for data, color, label in data_groups:
            rois_std = []
            for roi in rois:
                roi_std = 0
                for param in params:
                    # Calculate CV params
                    stds = data[data['ROI_name'] == roi][param].std()
                    roi_std += stds

                roi_std /= len(params)
                rois_std.append(roi_std)

            groups_rois_std[label] = rois_std
            plt.scatter(rois_labels, rois_std, color=color, label=label, s=70)
            plt.xticks(rois_labels, rotation='vertical', fontsize=20)
            plt.yticks(fontsize=16)

        for x, y1, y2 in zip(rois_labels, groups_rois_std['young'], groups_rois_std['old']):
            plt.plot([x, x], [y1, y2], color='gray', linestyle='--')

        plt.title('Average Std of all parameters', fontdict = {'fontsize' : 30})
        plt.ylabel('Average Std', fontdict = {'fontsize' : 20})
        plt.legend(fontsize=24, loc="upper left")

        if t_test_params:
            results = stats.ttest_ind(a=groups_rois_std[t_test_params[0]], b=groups_rois_std[t_test_params[1]])
            significance = 'significance' if results.pvalue <= 0.05 else 'no significance'
            p_val_str = constants.SUPERSCRIPTS[round(math.log10(results.pvalue))]
            plt.text(0.11, 0.90, f"$p < 10{p_val_str}$", fontsize=24, color='black', transform=plt.gca().transAxes)
            print(f't_test showed {significance} difference, p < 10{p_val_str}')

        return groups_rois_std

    @staticmethod
    def calculate_cv_f_test(data, group_by_param, params, x_axis):
        f_test_params = {}
        for param in params:
            f_test_params[param] = {}
            # Calculate CV params
            means = data.groupby(group_by_param)[[param, x_axis]].mean()
            stds = data.groupby(group_by_param)[param].std()
            cv = (stds / means[param])

            # split to two models
            simple_model = LinearRegression()
            x1 = np.array(means[param]).reshape(-1, 1)
            y1 = np.array(cv).reshape(-1, 1)
            simple_model.fit(x1, y1)

            complex_model = LinearRegression()
            x2 = np.array(means)
            y2 = np.array(cv).reshape(-1, 1)
            complex_model.fit(x2, y2)

            # Calculate the residual sum of squares (RSS) for each model
            rss1 = np.sum((y1 - simple_model.predict(x1)) ** 2)
            rss2 = np.sum((y2 - complex_model.predict(x2)) ** 2)

            f_test_params[param] = {
                "simple_model": {
                    "rss": rss1,
                    "p": 1,
                    "df": len(y1) - 2
                },
                "complex_model": {
                    "rss": rss2,
                    "p": 2,
                    "df": len(y2) - 3
                },
            }

        # check the f-test for each two models
        for param, param_values in f_test_params.items():
            group1_name, group1_params = list(param_values.items())[0]
            group2_name, group2_params = list(param_values.items())[1]

            # Compute the F-statistic
            f_statistic = ((group1_params['rss'] - group2_params['rss']) / (group2_params['p'] - group1_params['p'])) / \
                (group2_params['rss'] / group2_params['df'])
            # F = (group1_params['rss'] / group2_params['rss'])

            # Determine the critical value of the F-statistic
            alpha = 0.05  # Significance level
            # critical_value = f.ppf(1 - alpha, group1_params['df'], group2_params['df'])
            p_value = f.sf(
                f_statistic, group2_params['p'] - group1_params['p'], group2_params['df'])

            # Make a decision
            if p_value < alpha:
                print(
                    f"Param {param} - Reject the null hypothesis: {group2_name} is significantly better than {group1_name}")
            else:
                print(
                    f"Param {param} - Fail to reject the null hypothesis: No significant difference between {group1_name} and {group2_name}")


    @staticmethod
    def show_rois_differences_in_polar(groups, rois, params, titles, colors, method='mean', plot_cols=2):
        polar_data = []

        for group, title in zip(groups, titles):
            group_polar_data = []
            for roi, color in zip(rois, colors):
                method_data = getattr(group[group['ROI_name'] == roi][params], method)
                group_roi = pd.DataFrame([method_data()])

                group_polar_data.append(
                    {'group': group_roi, 'name': f'{roi}', 'color': color})

            polar_data.append(group_polar_data)
            
        PlotsManager.plot_rois_polar(polar_data, params, titles, plot_cols, plot_title=method)
