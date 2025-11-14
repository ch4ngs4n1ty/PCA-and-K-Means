import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    # read data
    data_frame = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv")
    
    # remove id column to get the data
    data_features = data_frame.iloc[:, 1:]
    
    # 2. compute cov matrix
    sigma = pd.DataFrame.cov(data_features)
    
    # compute eigenvector and eigenvals
    eigValues, eigVectors = eig(sigma)
    
    # 3. sort eigenval annd eigenvec highest to low abs
    sorted_indices = np.argsort(np.abs(eigValues))[::-1]
    sorted_eigenvalues = eigValues[sorted_indices]
    sorted_eigenvectors = eigVectors[:, sorted_indices]
    
    # 4. normalize eigenvals and plot
    normalized_eigenvalues = np.abs(sorted_eigenvalues) / np.sum(np.abs(sorted_eigenvalues))
    cumulative_sum = np.cumsum(normalized_eigenvalues)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cumulative_sum)), cumulative_sum, 'o-')
    plt.xlabel('Number of Eigenvectors')
    plt.ylabel('Cumulative Sum of Normalized Eigenvalues')
    plt.title('Cumulative Sum of Normalized Eigenvalues')
    plt.grid(True)
    plt.savefig('cumulative_eigenvalues.png')
    plt.show()
    
    # 5. print first 2 eigenvecs
    print("First Eigenvector (associated with largest eigenvalue):")
    print(sorted_eigenvectors[:, 0])
    print("\nSecond Eigenvector (associated with second largest eigenvalue):")
    print(sorted_eigenvectors[:, 1])
    
    # 6. project agglom data onto first 2 eigenvecs and plot
    first_two_eigenvectors = sorted_eigenvectors[:, :2]
    projected_data = np.dot(data_features.values, first_two_eigenvectors)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Projection onto First Two Principal Components')
    plt.grid(True)
    plt.savefig('pca_projection.png')
    plt.show()
    
    # 7. perform kmeans using package
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(projected_data)
    
    # plot kmeans
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projected_data[:, 0], projected_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Centroids')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('K-Means Clustering in PCA Space')
    plt.legend()
    plt.grid(True)
    plt.savefig('kmeans_clustering.png')
    plt.show()
    
    # 8. center of mass for eahc cluster
    cluster_centers = kmeans.cluster_centers_
    print("\nCluster Centers in PCA Space:")
    for i, center in enumerate(cluster_centers):
        print(f"Cluster {i}: {center}")
    
    feature_names = data_features.columns.tolist()
    
    # 9. reproject cluster centers back to original
    print("\nRe-projected Cluster Centers back to Original Feature Space:")
    for i, center in enumerate(cluster_centers):
        reprojected = np.dot(center, first_two_eigenvectors.T)
        print(f"\nCluster {i}:")
        for j, feature_name in enumerate(feature_names):
            print(f"  {feature_name}: {reprojected[j]:.2f}")

if __name__ == "__main__":
    main()