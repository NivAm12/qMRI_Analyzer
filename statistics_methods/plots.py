import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import constants
from collections import Counter
import nibabel as nib
import copy
import scipy.ndimage as ndi
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PlotsManager:
    @staticmethod
    def plot_heatmap(data: pd.DataFrame, title: str, project_name: str):
        sns.set(font_scale=0.5)
        plt.figure(figsize=(20, 10))
        cluster_map = sns.heatmap(data, linewidth=.5, cmap='coolwarm')
        plt.title(f'{title}')

        if project_name:
            wandb_run = wandb.init(
                project=project_name,
                name=f'{title}'
            )

            wandb_run.log({f'{title}': wandb.Image(plt)})
            wandb_run.finish()
            plt.close()
        else:
            plt.show()


    @staticmethod
    def create_and_plot_dendrogram(clusters, labels, title, linkage_metric, project_name=None, figsize=(20, 10)):
        plt.figure(figsize=figsize)
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

        else:
            plt.show()

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

    @staticmethod
    def plot_colors_on_brain(example_subject: str, rois_color: pd.Series, rois_dict: dict,
                              title: str, color_type: str):
        rois_values = list(rois_dict.keys())
        flipped_roi_dict = {value: key for key, value in rois_dict.items()}
        data_path = os.path.join(constants.ANALYSIS_DIR, example_subject)
        seg_path = os.path.join(data_path, os.listdir(data_path)[0], constants.BASIC_SEG)
        brain_path = os.path.join(data_path, os.listdir(data_path)[0], constants.MAP_TV)
        save_path = os.path.join(constants.CLUSTERING_PATH, f'lut_{title}.nii.gz')

        # read the map
        seg_file = nib.load(seg_path)
        seg_file_data = seg_file.get_fdata()
        brain_file_data = nib.load(brain_path).get_fdata()
        color_map = copy.deepcopy(seg_file_data)

        # paint each roi with his cluster color
        roi_values_as_other_type = np.array(list(rois_values), dtype=seg_file_data.dtype)
        remove_mask = np.logical_not(np.isin(seg_file_data, roi_values_as_other_type))

        for roi, roi_color in rois_color.items():
            roi_mask = np.where(seg_file_data == flipped_roi_dict[roi])
            color_map[roi_mask] = roi_color

        # save and show the map
        tr = min(rois_color)
        color_map[remove_mask] = -4
        color_map = nib.Nifti1Image(color_map, seg_file.affine)

        nib.save(color_map, save_path)
        os.system(f'freeview -v {brain_path} {save_path}:colormap={color_type}')
        # plt.figure(figsize = (5, 5))
        # plt.grid(False)

        # slice = 100
        # plt.imshow(ndi.rotate(brain_file_data[slice], 90), cmap='gray')
        # plt.imshow(ndi.rotate(color_map.get_fdata()[slice], 90), cmap='hot', alpha=0.5, vmin=-1)

    @staticmethod
    def plot_rois_polar(data1, data2, thetas, range, titles):
        fig = make_subplots(rows=1, cols=2, subplot_titles=titles, specs=[[{"type": "polar"}, {"type": "polar"}]])

        fig.add_trace(go.Scatterpolar(
            r=data1[0]['r'],
            theta=thetas,
            opacity = 0.7,
            fill='toself',
            fillcolor = 'red',
            name=data1[0]['name']),
            row=1, col=1)
        fig.add_trace(go.Scatterpolar(
            r=data1[1]['r'],
            theta=thetas,
            opacity = 0.5,
            fill='toself',
            fillcolor = 'blue',
            name=data1[1]['name']),
            row=1, col=1)
        fig.add_trace(go.Scatterpolar(
            r=data2[0]['r'],
            theta=thetas,
            opacity = 0.7,
            fill='toself',
            fillcolor = 'red',
            name=data2[0]['name']),
            row=1, col=2)
        fig.add_trace(go.Scatterpolar(
            r=data2[1]['r'],
            theta=thetas,
            opacity = 0.5,
            fill='toself',
            fillcolor = 'blue',
            name=data2[1]['name']),
            row=1, col=2)
        
        fig.layout['polar'].update(dict(
            radialaxis=dict(
            visible=True,
            # range=range
            )))
        
        fig.layout['polar2'].update(dict(
            radialaxis=dict(
            visible=True,
            # range=range
            )))

        fig.show()