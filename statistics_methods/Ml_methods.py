import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from scipy import stats


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
    
    @staticmethod
    def spectral_clustering(similarity_matrix: pd.DataFrame, n_clusters: int):
        # Compute the Laplacian matrix
        similarity_matrix_np = similarity_matrix.to_numpy()
        np.fill_diagonal(similarity_matrix_np, 0)
        similarity_matrix_np[similarity_matrix_np <= 0] = 0

        spectral_clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )

        y_spectral = spectral_clustering.fit_predict(similarity_matrix_np)

        # Compute the Laplacian matrix
        laplacian = np.diag(np.sum(similarity_matrix_np, axis=1)) - similarity_matrix_np

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        # Select the eigenvectors corresponding to the smallest non-zero eigenvalues
        eigenvectors_k = eigenvectors[:, 1:n_clusters]
        z_scores = np.abs(stats.zscore(eigenvectors_k, axis=0))

        # Define a threshold for identifying outliers
        threshold = 2

        # Create a boolean mask to filter out outliers
        mask = (z_scores < threshold).all(axis=1)
        eigenvectors_k = eigenvectors_k[mask]
        y_spectral = y_spectral[mask]
    
        # Plot the selected eigenvectors
        plt.figure(figsize=(12, 6))
        plt.scatter(eigenvectors_k[:, 0], eigenvectors_k[:, 1], c=y_spectral, cmap='viridis')
        plt.xlabel('Eigenvector 1')
        plt.ylabel('Eigenvector 2')
        plt.title('Visualization of Eigenvectors')
        plt.show()

        clusters = {}

        for index, label in zip(similarity_matrix.index, y_spectral):
            if label not in clusters.keys():
                clusters[label] = []
            clusters[label].append(index)

        return clusters, y_spectral

    @staticmethod
    def evaluate_clusters(similarity_matrix: pd.DataFrame, max_clusters: int):
        silhouette_scores = []
        davies_bouldin_scores = []
        similarity_matrix_np = similarity_matrix.to_numpy()
        similarity_matrix_np[similarity_matrix_np <= 0] = 0

        for n_clusters in range(2, max_clusters + 1):
            _, labels = Ml_methods.spectral_clustering(similarity_matrix, n_clusters)
            silhouette_avg = silhouette_score(similarity_matrix_np, labels, metric='precomputed')
            davies_bouldin_avg = davies_bouldin_score(similarity_matrix_np, labels)
            
            silhouette_scores.append((n_clusters, silhouette_avg))
            davies_bouldin_scores.append((n_clusters, davies_bouldin_avg))
            
            print(f'Clusters: {n_clusters}, Silhouette Score: {silhouette_avg}, Davies-Bouldin Index: {davies_bouldin_avg}')


        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.plot(np.array(silhouette_scores)[:, 0], np.array(silhouette_scores)[:, 1], marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores for Different Clusters')

        plt.subplot(1, 2, 2)
        plt.plot(np.array(davies_bouldin_scores)[:, 0], np.array(davies_bouldin_scores)[:, 1], marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Davies-Bouldin Index')
        plt.title('Davies-Bouldin Index for Different Clusters')

        plt.show()
            