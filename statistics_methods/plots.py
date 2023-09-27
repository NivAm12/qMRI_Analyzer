import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import constants
from collections import Counter


class PlotsManager:
    @staticmethod
    def plot_heatmap(data: pd.DataFrame, group_title: str, project_name: str):
        sns.set(font_scale=0.5)
        plt.figure(figsize=(20, 10))
        cluster_map = sns.heatmap(data, linewidth=.5, cmap='coolwarm')
        plt.title(f'Correlations of {group_title} group')

        if project_name:
            wandb_run = wandb.init(
                project=project_name,
                name=f'{group_title} Correlations'
            )

            wandb_run.log({f'{group_title}': wandb.Image(plt)})
            wandb_run.finish()

        plt.close()

    @staticmethod
    def create_and_plot_dendrogram(clusters, labels, title, linkage_metric, project_name=None):
        plt.figure(figsize=(20, 10))
        dendrogram_data = dendrogram(clusters, labels=labels,
                                     orientation='right', leaf_font_size=8)
        plt.title(f'Hierarchical Clustering Dendrogram of {title} group with {linkage_metric}')
        plt.ylabel('ROI')
        plt.xlabel('Distance')

        if project_name:
            wandb_run = wandb.init(
                project=project_name,
                name=f'{title} hierarchical clustering {linkage_metric}',
                config={
                    'linkage_metric': linkage_metric
                }
            )

            wandb_run.log({f'{title}': wandb.Image(plt)})
            wandb_run.finish()

        plt.close()

        return dendrogram_data

    @staticmethod
    def create_clusters_color_list(clusters, distance):
        color_new_list = []
        clusters_map = fcluster(clusters, distance, 'distance')
        clusters_counter = Counter(clusters_map)

        # Sort the dictionary items by their values in descending order
        counter_values = sorted(clusters_counter.items(), key=lambda x: x[1], reverse=True)
        counter_values = [item[0] for item in counter_values]

        for color in clusters_map:
            color_new_list.append(counter_values.index(color) + 10)

        return color_new_list

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
                        plt.title(f"{subject}\n info of {params} per {constants.SUB_CORTEX_DICT[rois[i]]} \n and "
                                  f"{constants.SUB_CORTEX_DICT[rois[j]]}")
                        plt.ylabel(constants.SUB_CORTEX_DICT[rois[j]])
                        plt.xlabel(constants.SUB_CORTEX_DICT[rois[i]])
                        if save_address:
                            plt.savefig(save_address_for_func + "/" +
                                        f"cor_{constants.SUB_CORTEX_DICT[rois[i]]}_and_{constants.SUB_CORTEX_DICT[rois[j]]}.png")
                        plt.show()
