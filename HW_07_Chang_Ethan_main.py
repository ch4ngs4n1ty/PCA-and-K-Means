# Ethan Chang - CSCI 420 HW 7
#

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig
from sklearn.cluster import KMeans

def main():

    df = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv")

    df = df.drop(df.columns[0], axis=1) # Drop the guest id

    feature_names = df.columns.to_list()


    # Covariance and eigendecomposition

    cov_matrix = df.cov().values # Covariance Matrix (20 x 20)

    #print(sigma)

    eigvals, eigvecs = eig( cov_matrix )

    eigvals = np.real_if_close(eigvals)
    eigvecs = np.real_if_close(eigvecs)

    idx = np.argsort(-np.abs(eigvals))
    sorted_eigvals = eigvals[idx]
    sorted_eigvecs = eigvecs[:, idx]

    #print(eig_array)

    total = np.sum(np.abs(sorted_eigvals))

    normalized = np.abs(sorted_eigvals) / total

    cumulative = np.cumsum(normalized)

    plt.plot(range(1, len(cumulative) + 1), cumulative, marker='o')
    plt.xlabel('Number of Eigenvalues')
    plt.ylabel('Cumulative Sum (Normalized)')
    plt.title('Cumulative Sum of Normalized Eigenvalues')
    plt.grid(True)
    
    plt.savefig("cumulative_plot.png", dpi=300, bbox_inches='tight')

    # First two eigenvectors that prints 1x20 rows
    first_two_vec = sorted_eigvecs[:, :2].T
    print("First two eigenvectors (rows):")
    print(first_two_vec)

    # Project original data onto the first two eigenvectors 
    X = df.values

    X_centered = X - X.mean(axis=0, keepdims=True)   # center the data

    top2_vecs = sorted_eigvecs[:, :2]                # shape (20, 2)

    projected_2d = np.dot(X_centered, top2_vecs)         # shape (N, 2)

    # Eigen value variances
    evr1 = sorted_eigvals[0] / sorted_eigvals.sum()
    evr2 = sorted_eigvals[1] / sorted_eigvals.sum()

    # 2D Scatter Plot of Clusters
    plt.figure()
    plt.scatter(projected_2d[:, 0], projected_2d[:, 1], s=20, alpha=0.85)
    plt.xlabel(f'PC1 ({evr1:.1%} var)')
    plt.ylabel(f'PC2 ({evr2:.1%} var)')
    plt.title('Projection onto First Two Eigenvectors (2D Scatter)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axis('equal')
    plt.savefig('pca_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()


    # Quick attribute importance for each PC: largest |loading|
    loadings_df = pd.DataFrame(first_two_vec, 
                               columns=feature_names, 
                               index=["PC1","PC2"])

    # Outputs the two vectors
    # PC1 is dominated by horror, classics, non-fiction, romance, and games
    # PC2 is dominated by baby toddler, teen, self imporvement, games, and non-fiction
    for pc in ["PC1","PC2"]:
        top = loadings_df.loc[pc].abs().sort_values(ascending=False)
        print(f"\nTop 5 attributes for {pc} by |loading|:")
        print(top.head(5))


    #vector_matrix = pd.DataFrame(first_two_vec)


if __name__ == "__main__":
    main()