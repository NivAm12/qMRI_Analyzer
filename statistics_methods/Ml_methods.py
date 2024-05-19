import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd


class Ml_methods:

    @staticmethod
    def kmeans_cluster(data_groups, params, label, k):
        clustered_data = {}

        # cluster each group data
        for data, name in data_groups:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data[params])

            cluster_labels = kmeans.labels_

            pca = PCA(n_components=2)
            pca_data = Ml_methods.reduce_dim(pca, data, params, label)
            pca_data['cluster_label'] = cluster_labels

            for label_name, label_data in pca_data.groupby('label'):
                # create a list of this label data for all the groups
                if label_name not in clustered_data.keys():
                    clustered_data[label_name] = []

                clustered_data[label_name].append(
                    {'data': label_data, 'name': name})

        for label_name, label_data in clustered_data.items():
            fig, ax = plt.subplots(nrows=1, ncols=len(
                label_data), figsize=(18, 6))
            fig.suptitle(f'k-means Clustering with PCA ,{label_name}')

            for col, group_clustered_data in enumerate(label_data):
                data = group_clustered_data['data']
                name = group_clustered_data['name']

                sns.scatterplot(x=data['var_1'], y=data['var_2'],
                                hue=data['cluster_label'], palette='viridis', ax=ax[col])
                ax[col].set_title(
                    f'{name}')

        plt.show()

    @staticmethod
    def choose_k_for_kmeans(data):
        silhouette_scores = []

        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)

        plt.figure(figsize=(8, 4))
        plt.plot(range(2, 11), silhouette_scores, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Optimal k')
        plt.show()

        return silhouette_scores

    @staticmethod
    def reduce_dim(model, data, params, label=None):
        reduced_data = model.fit_transform(data[params])

        # Generate dynamic column names
        column_names = [f"var_{i + 1}" for i in range(reduced_data.shape[1])]
        reduced_data = pd.DataFrame(
            reduced_data, columns=column_names, index=data.index)

        if label:
            reduced_data['label'] = data[label]

        return reduced_data
