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
from nilearn import plotting
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from nilearn import plotting, surface
import matplotlib.ticker as ticker
import networkx as nx


# COLORS_MAPS
# Define a custom colormap
coolwarm_colors = [
    (0.0, 'blue'),    # Start (lowest values)
    (0.5, 'gray'),    # Midpoint (zero value)
    (1.0, 'red')      # End (highest values)
]

custom_cmap_coolwarm = LinearSegmentedColormap.from_list(
    'custom_cmap', coolwarm_colors)


class PlotsManager:
    @staticmethod
    def plot_heatmap(data: pd.DataFrame, title: str, project_name: str=None):
        sns.set(font_scale=0.5)
        plt.figure(figsize=(20, 10))
        cluster_map = sns.heatmap(data, linewidth=.5, cmap='coolwarm')
        plt.title(f'{title}', fontsize=12)

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
    def create_and_plot_dendrogram(clusters, labels, title, linkage_metric, project_name=None, figsize=(20, 12), show=True):
        plt.figure(figsize=figsize)
        dendrogram_data = dendrogram(clusters, labels=labels,
                                     orientation='right', leaf_font_size=8)
        plt.title(
            f'Hierarchical Clustering of {title} group with {linkage_metric} linkage', fontsize=12)
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

        if show:
            plt.show()

        return dendrogram_data

    @staticmethod
    def create_clusters_color_list(clusters, distance):
        color_new_list = []
        clusters_map = fcluster(clusters, distance, 'distance')
        clusters_counter = Counter(clusters_map)

        # Sort the dictionary items by their values in descending order
        counter_values = sorted(clusters_counter.items(),
                                key=lambda x: x[1], reverse=True)
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
                        plt.legend(all_scatter_plots, params,
                                   scatterpoints=1, ncol=3, fontsize=8)
                        plt.title(f"{subject}\n info of {params} per {constants.SUB_CORTEX_DICT[rois[i]]} \n and "
                                  f"{constants.SUB_CORTEX_DICT[rois[j]]}")
                        plt.ylabel(constants.SUB_CORTEX_DICT[rois[j]])
                        plt.xlabel(constants.SUB_CORTEX_DICT[rois[i]])
                        if save_address:
                            plt.savefig(save_address_for_func + "/" +
                                        f"cor_{constants.SUB_CORTEX_DICT[rois[i]]}_and_{constants.SUB_CORTEX_DICT[rois[j]]}.png")
                        plt.show()

    @staticmethod
    def plot_colors_on_brain2(rois_color: pd.Series, prefix: str, title: str,
                              lh_annot_path: str, rh_annot_path: str):
        hemis = [
            {'name': 'lh', 'annot_path': lh_annot_path,
                'surf': constants.EXAMPLE_SURFACE_PIAL_LH_PATH, 'hemi': 'left'},
            {'name': 'rh', 'annot_path': rh_annot_path,
                'surf': constants.EXAMPLE_SURFACE_PIAL_RH_PATH, 'hemi': 'right'}
        ]

        cmap = plt.get_cmap('coolwarm')
        fig, ax = plt.subplots(
            1, 2, subplot_kw={'projection': '3d'}, figsize=(15, 5))

        for col, hemi in enumerate(hemis):
            labels, ctab, names = nib.freesurfer.read_annot(hemi['annot_path'])
            labels_decoded = np.array(
                [f"{prefix}{hemi['name']}-{name.decode()}" for name in names])

            # surface_data = np.zeros(len(labels))
            surface_data = np.full(len(labels), np.nan)
            
            for roi_name, roi_val in rois_color.items():
                if hemi['name'] not in roi_name:
                    continue

                label_index = np.where(labels_decoded == roi_name)[0][0]
                surface_data[labels == label_index] = roi_val

            # Load the surface mesh
            pial_mesh = surface.load_surf_mesh(hemi['surf'])
            # save_path = f"{constants.CLUSTERING_PATH}/{hemi['name']}_{title}.pial"
            curvature_path = f"{constants.CLUSTERING_PATH}/{hemi['name']}_{title}.curv"
            nib.freesurfer.write_morph_data(curvature_path, surface_data)
            print(f'curvature saved at {curvature_path}')

            # Plot the surface with the data
            surf = plotting.plot_surf_roi(
                pial_mesh, roi_map=surface_data, hemi=hemi['hemi'], cmap=cmap, bg_map=None,
                title=f'{title} - {hemi["name"]} hemi', axes=ax[col], colorbar=False,
                view='lateral',
                threshold=None,
                title_font_size=35,
                vmin=-0.4, vmax=0.8
            )  

          # Adding a color bar manually
            norm = plt.Normalize(vmin=-0.4, vmax=0.8)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax[col], orientation='vertical')
            cbar.set_label('Value')  # Optional: Add a label to the color bar

        # Plot the original surface with ROIs
        # plotting.plot_surf_roi(
        #     pial_mesh, roi_map=labels, hemi=hemi['hemi'],
        #     view='lateral', cmap='tab20', bg_map=None,
        #     colorbar=True,
        #     title=f'Original ROIs - {hemi["name"]}', axes=ax[col+1]
        # )

        plt.show()

    @staticmethod
    def plot_colors_on_brain(rois_color: pd.Series, prefix: str, title: str):
        hemis = [{'name': 'lh', 'path': constants.EXAMPLE_ANNOT_LH_PATH}, {
            'name': 'rh', 'path': constants.EXAMPLE_ANNOT_RH_PATH}]

        # Create a colormap for the values
        cmap = plt.get_cmap('seismic')
        norm = plt.Normalize(vmin=-0.4, vmax=0.8)

        for hemi in hemis:
            labels, ctab, names = nib.freesurfer.read_annot(hemi['path'])
            labels_decoded = np.array(
                [f"{prefix}{hemi['name']}-{name.decode()}" for name in names])

            for roi_name, roi_val in rois_color.items():
                if hemi['name'] not in roi_name:
                    continue

                label_index = np.where(labels_decoded == roi_name)[0][0]
                # Convert the correlation value to an RGB tuple using the colormap
                rgb = cmap(norm(roi_val))[:3]
                # Convert RGB to BGR for FreeSurfer compatibility and add alpha channel
                bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255), 255)

                # Replace the color in the color table
                ctab[label_index, :4] = bgr

            # Save the updated annotation file
            save_path = f"{constants.CLUSTERING_PATH}/{hemi['name']}_{title}.annot"
            nib.freesurfer.write_annot(save_path, labels, ctab, names)
            print(f'annot file saved at\n {save_path}')

    @staticmethod
    def plot_rois_polar(data, thetas, sub_titles, cols, plot_title):
        fig = make_subplots(rows=1, cols=cols, subplot_titles=sub_titles if cols > 1 else [plot_title], specs=[
                            [{"type": "polar"}]*cols])

        for col, polar_group in enumerate(data):
            for roi_group in polar_group:
                for _, subject_roi in roi_group['group'].iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=subject_roi.to_numpy(),
                        theta=thetas,
                        fill='toself',
                        line_color=roi_group['color'],
                        name=roi_group['name']),
                        row=1, col=(col+1 if cols > 1 else 1))

        for annotation in fig['layout']['annotations']: 
            annotation['yanchor']='top'
            annotation['y']=1.2
    
        fig.layout['polar'].update(dict(
            radialaxis=dict(
                visible=True,
            )))
        if cols > 1:
            fig.layout['polar2'].update(dict(
                radialaxis=dict(
                    visible=True,
                )))

        fig.show()

    # @staticmethod
    # def plot_std_for_rois_by_params(data_groups, params, fig_size=(20, 8)):
    #     fig, ax = plt.subplots(nrows=len(params), figsize=(fig_size[0], fig_size[1] * len(params)))
    #     markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']

    #     for data, color, label in data_groups:
    #         std_df = data.groupby('ROI_name')[params].std()

    #         for i, (param, marker) in enumerate(zip(params, markers)):
    #             ax[i].scatter(std_df.index, std_df[param], color=color, label=f'{label} {param}', marker=marker)

    #             ax[i].set_xticks(std_df.index)
    #             ax[i].set_xticklabels(std_df.index, rotation='vertical', fontsize=10)
    #             ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)
    #             ax[i].legend()

    #     fig.tight_layout()


    @staticmethod
    def plot_std_for_rois_by_params(data_groups, params, fig_size=(20, 8)):
        fig, ax = plt.subplots(nrows=len(params), figsize=(fig_size[0], fig_size[1] * len(params)))
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']

        previous_values = {param: {} for param in params}

        for data, color, label in data_groups:
            std_df = data.groupby('ROI_name')[params].std()

            for i, (param, marker) in enumerate(zip(params, markers)):
                # Scatter plot
                ax[i].scatter(std_df.index, std_df[param], color=color, label=f'{label}', marker=marker)
                
                # Plot vertical lines to connect groups
                for roi in std_df.index:
                    if roi in previous_values[param]:
                        ax[i].plot([roi, roi], [previous_values[param][roi], std_df[param][roi]], color='gray', linestyle='dotted', linewidth=2)
                
                # Update previous values
                for roi in std_df.index:
                    previous_values[param][roi] = std_df[param][roi]
                
                ax[i].set_xticks(std_df.index)
                ax[i].set_xticklabels(std_df.index, rotation='vertical', fontsize=10)
                ax[i].grid(False)
                ax[i].set_title(f'{param.upper()} std')
                ax[i].legend()

        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_similarity_graph(similarity_matrix: pd.DataFrame, labels: np.ndarray):
        G = nx.Graph()
    
        # Add nodes with labels
        for i in range(len(similarity_matrix)):
            G.add_node(i, label=similarity_matrix.index[i])  # Use DataFrame index as label
        
        # Add edges with weights
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix.iloc[i, j] < 0.2:  # Add edge only for positive similarity
                    G.add_edge(i, j, weight=similarity_matrix.iloc[i, j])
        
        # Get positions for the nodes using a layout algorithm
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the nodes with colors based on the cluster labels
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))
        
        node_colors = [color_map[labels[node]] for node in G.nodes()]
        
        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=10)  # Use custom labels
        
        plt.title('Similarity Graph with Spectral Clustering Labels')
        plt.show()
