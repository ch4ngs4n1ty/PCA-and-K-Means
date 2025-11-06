# Ethan Chang - CSCI 420 HW 7
#

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def main():

    df = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv")

    df = df.drop(df.columns[0], axis=1) # Drop the guest id

    cov_matrix = pd.DataFrame.cov(df) # Covariance Matrix (20 x 20)

    #print(sigma)

if __name__ == "__main__":
    main()