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
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import networkx as nx


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
        similarity_matrix_np[similarity_matrix_np < 0] = 0

        spectral_clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )

        labels = spectral_clustering.fit_predict(similarity_matrix_np)

        clusters = {}

        for index, label in zip(similarity_matrix.index, labels):
            if label not in clusters.keys():
                clusters[label] = []
            clusters[label].append(index)

        return clusters

        

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