# Ethan Chang and Jacky Chan - CSCI 420 HW 7
# File: HW_07_Chang_Ethan_main.py
# Compile: python HW_07_Chang_Ethan_main.py

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig
from sklearn.cluster import KMeans

"""
The main function of the program.

Initializes the data by loading it from the csv file and cleaning it.
Starting off with computing eigenvectors and eigenvalues.

Start computing K-Means in 2-D PCA space and visualize clusters and centers.
"""
def main():

    #####################################################################################################
    # Data Initialization and building eigenvalue and eigenvector arrays
    #####################################################################################################

    df = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv") # Reads the csv file

    df = df.drop(df.columns[0], axis=1) # Drop the guest id

    feature_names = df.columns.to_list() # Column names into array (20 shopping categories)

    # Covariance and eigendecomposition
    cov_matrix = df.cov().values # Covariance Matrix (20 x 20)

    # Computing eigenvalues and eigenvectors of covariance matrix
    eigvals, eigvecs = eig( cov_matrix ) 


    # Sort eigenvalues in descending order of absolute value
    # idx is an array of indicies that sort -[eigvals] in ascending order
    idx = np.argsort(-np.abs(eigvals))

    # Integrate idx into the eigvalues and eigvectors to ascending order
    sorted_eigvals = eigvals[idx]
    sorted_eigvecs = eigvecs[:, idx]

    # Compute total value (sum of absolute eigenvalues) for normalization
    total = np.sum(np.abs(sorted_eigvals))

    # Normalize eigenvalues by total to get fraction of variance
    normalized = np.abs(sorted_eigvals) / total

    # Compute cumulative sum of variance
    # cumulative[i] = sum of normalized eigenvalues from 1 to i + 1
    cumulative = np.cumsum(normalized)

    #####################################################################################################
    # Plot: The Cumulative Normalized Eigenvalues
    #####################################################################################################

    # Plot of cumulative sum
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker='o')
    plt.xlabel('Number of Eigenvalues')
    plt.ylabel('Cumulative Sum (Normalized)')
    plt.title('Cumulative Sum of Normalized Eigenvalues')
    plt.grid(True) # shows grid for readability
    
    # Save figure of the plot
    plt.savefig("cumulative_plot.png", dpi=300, bbox_inches='tight')

    #####################################################################################################
    # Take first two eigenvectors (2D Vectors)
    #####################################################################################################

    # First two eigenvectors that prints 1x20 rows
    # This vector has a shape of (2, 20) and each row is one eigenvector
    first_two_vec = sorted_eigvecs[:, :2].T
    print("First two eigenvectors (rows):")
    print(first_two_vec)

    X = df.values # Original data set excluding GuestID (converts DataFrame to NumPy array)

    X_centered = X - X.mean(axis=0, keepdims=True)   # Mean-centered version of X (new_data = data - all_means)

    top2_vecs = sorted_eigvecs[:, :2] # First two eigenvectors, each column = one eigenvector

    # projected 2D has (number of samples, 2)
    projected_2d = np.dot(X_centered, top2_vecs) # 2D data, each row is one sample's coordinates in PC1 to PC2 space

    #####################################################################################################
    # 2D Scatter Plot of data in PCA Space
    #####################################################################################################

    # Eigenvalue variances ratio for first and second componenets
    evr1 = sorted_eigvals[0] / sorted_eigvals.sum()
    evr2 = sorted_eigvals[1] / sorted_eigvals.sum()

    # 2D Scatter Plot of PCA
    plt.figure()

    # Scatter plot of projected points in 2D PCA space
    plt.scatter(projected_2d[:, 0], projected_2d[:, 1], s=20, alpha=0.85)

    plt.xlabel(f'PC1 ({evr1:.1%} var)') # Label PC1 by variance percentage
    plt.ylabel(f'PC2 ({evr2:.1%} var)') # Label PC2 by variance percentage
    plt.title('First Two Eigenvectors (2D Scatter)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axis('equal')
    plt.savefig('pca_scatter.png', dpi=300, bbox_inches='tight')

    #####################################################################################################
    # Loadings for PC1 and PC2
    #####################################################################################################

    # Quick attribute importance for each PC: largest |loading|
    # Create a dataframe (PC1, PC2) and columns are the 20 features
    loadings_df = pd.DataFrame(first_two_vec, 
                               columns=feature_names, 
                               index=["PC1","PC2"])

    # Outputs the two vectors
    # PC1 is dominated by horror, classics, non-fiction, romance, and games
    # PC2 is dominated by baby toddler, teen, self imporvement, games, and non-fiction
    # Print five features with largest absolute loading
    for pc in ["PC1","PC2"]:
        top = loadings_df.loc[pc].abs().sort_values(ascending=False)
        print(f"\nTop 5 attributes for {pc} by |loading|:")
        print(top.head(5))

    
    #####################################################################################################
    # K-Means Clustering in 2-D PCA Space
    #####################################################################################################

    k = 4 # There are 4 clusters coming from the scatter plot of 2D data

    # Create KMeans object with k clusters with fixed random state
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit k-means to 2-D data and get cluster labels for each sample
    labels = kmeans.fit_predict(projected_2d)


    plt.figure(figsize=(7, 6))

    # K-Means Figure For Cluster Plot
    plt.scatter(projected_2d[:, 0], projected_2d[:, 1],
                c=labels, cmap='viridis', s=30, alpha=0.8, edgecolors='k')
    
    # Plot K-Mean cluster centers in 2-D PCA space by using red X markers
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='X', s=200, label='Cluster Centers')
    
    plt.title(f'K-Means Clustering (k={k}) on PCA Projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'kmeans_clusters_k{k}.png', dpi=300, bbox_inches='tight')

    #####################################################################################################
    # Print Cluster Centers In PCA Space
    #####################################################################################################
    centers = kmeans.cluster_centers_

    print("\nCluster Centers (in PCA 2D space):")

    # Loop through each cluster center and print PC1 and PC2 coordinates
    for i, c in enumerate(centers):
        print(f"Cluster {i}: PC1 = {c[0]:.4f}, PC2 = {c[1]:.4f}")


    #####################################################################################################
    # Re-project cluster centers back to orginal feature space
    #####################################################################################################

    # Re-Projection: Multiply 2-D centers by top2_vecs so it equals into 4 x 20
    reprojected = np.dot(centers, top2_vecs.T)   # shape (4, 20)

    # Put reprojected cluster centers into the data frame
    reproj_df = pd.DataFrame(reprojected.round(), columns=feature_names)

    print("\nReprojected cluster centers back to original feature space:")

    # Save reprojected cluster centers as CSV
    reproj_df.to_csv("reprojected_cluster_centers.csv", index=False)

if __name__ == "__main__":
    main()