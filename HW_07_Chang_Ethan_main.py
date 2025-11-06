# Ethan Chang - CSCI 420 HW 7
#

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def main():

    df = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv")

    new_matrix = np.delete(df, 0, axis=1)

    print(new_matrix)




    sigma = pd.DataFrame.cov(df)

    #print(sigma)

if __name__ == "__main__":
    main()