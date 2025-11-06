# Ethan Chang - CSCI 420 HW 7
#

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig


def main():

    df = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv")

    df = df.drop(df.columns[0], axis=1) # Drop the guest id

    cov_matrix = pd.DataFrame.cov(df) # Covariance Matrix (20 x 20)

    #print(sigma)

    eigValues, eigVECTORS = eig( cov_matrix )

    val_matrix = np.matrix(eigValues)

    vector_matrix = np.matrix(eigVECTORS)

if __name__ == "__main__":
    main()