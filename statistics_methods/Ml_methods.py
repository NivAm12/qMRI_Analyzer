import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd


class Ml_methods:

    @staticmethod
    def kmeans_cluster(data, params, label, k):        
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data[params])

        cluster_labels = kmeans.labels_

        pca = PCA(n_components=2)
        pca_data = Ml_methods.reduce_dim(pca, data, params, label)
        pca_data['cluster_label'] = cluster_labels

        for label_name, label_data in pca_data.groupby('label'):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=label_data['var_1'], y=label_data['var_2'], hue=label_data['cluster_label'], palette='viridis')
            plt.title(f'k-means Clustering with PCA Projection, {label_name}')
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

        column_names = [f"var_{i + 1}" for i in range(reduced_data.shape[1])]  # Generate dynamic column names
        reduced_data = pd.DataFrame(reduced_data, columns=column_names, index=data.index)

        if label:
            reduced_data['label'] = data[label]

        return reduced_data