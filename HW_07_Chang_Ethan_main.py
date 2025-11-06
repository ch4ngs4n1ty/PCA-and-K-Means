# Ethan Chang - CSCI 420 HW 7
#

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig


def main():

    df = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv")

    df = df.drop(df.columns[0], axis=1) # Drop the guest id

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
    plt.close()

    first_two_vec = sorted_eigvecs[:, :2].T
    print("First two eigenvectors (rows):")
    print(first_two_vec)

    #vector_matrix = pd.DataFrame(first_two_vec)


if __name__ == "__main__":
    main()