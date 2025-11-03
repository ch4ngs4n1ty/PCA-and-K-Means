#
#
#  QUICK PYTHON PROGRAM TO: test the covariance.
#
#
import sys
import datetime
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt


def main() :

    print("Examples from https://realpython.com/pandas-dataframe/ ")

    ##############################################################################
    #
    #  Read in the data using Pandas:
    #
    print("Reading in the file : Two_Dimensional_Data_For_Students_To_Check.csv")
    data_frame = pd.read_csv("Two_Dimensional_Data_For_Students_To_Check.csv")
    print("\n")

    # Playing with Pandas here:
    print(".ndim = ")
    print(data_frame.ndim)

    # Playing with Pandas here:
    print(".shape = ")
    print(data_frame.shape)

    # Playing with Pandas here:
    print(".size = ")
    print(data_frame.size)
    print("\n\n")

    # Playing with Pandas here:
    print("data_frame.index = ");
    print(data_frame.index);
    print("\n")

    # Playing with Pandas here:
    print("data_frame.columns = ");
    print(data_frame.columns);
    print("\n")

    ##############################################################################
    #
    #  Here we find the covariance of the data:
    #
    sigma = pd.DataFrame.cov( data_frame )
    print("Sigma Found Using Python = ");
    print( sigma )

    print("  ")   # Blank line
    print( "Dr K thinks that the Sigma Should be = " );
    print("        X          Y")
    print("X    8.487386 -7.520122")
    print("Y   -7.520122  8.501368")

    print("  ")   # Blank line
    print( "Matlab thinks that the Sigma Should be = " );
    print("       X          Y")
    print("X    8.4874   -7.5201")
    print("Y   -7.5201    8.5014")

    print("")
    eigValues, eigVECTORS = eig( sigma )

    print( "Eigen VECTORS =" )
    print( np.matrix( eigVECTORS ) )

    print(" ")
    print( "Eigen VALUES =" )
    print( np.matrix( eigValues ) )

    print(" ")
    print( "Data Frame = " )
    print( data_frame )
    print( "Data Frame.Keys = " )
    print( data_frame.keys )

    print( 'Example data : ' )
    row2 = data_frame.iloc[1] 
    print("row2 = ", row2 )
    col_1 = data_frame['LeftColumn']	# Returns a "Series"
    col_2 = data_frame["RightColumn"]	# Returns a "Series"

    print("\n Trying a Plotting Routine:")
    # data_frame.plot(data_frame["LeftColumn"], data_frame["RightColumn"], style="o")
    plt.figure(figsize=(7, 5))
    plt.plot( col_1, col_2, '.' )
    plt.show()


#
#  Boiler Plate -- standard code here:
#
if ( __name__ == "__main__" ) :
    main()
else:
    print("this is NOT main ... something went wrong.")


