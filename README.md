# Cross-Study Matrix Completion via a Random Forest Algorithm
**Authors:** Tal Einav and Rong Ma
<br>
https://doi.org/10.1016/j.crmeth.2023.100540
<br/>
[![DOI](https://zenodo.org/badge/489141368.svg)](https://zenodo.org/badge/latestdoi/489141368)

A random forest algorithm for matrix completion across multiple studies.

For a collection of datasets generated from multiple studies, where the measured features may have little-to-no overlap across the studies, our method provides predictions along with uncertainty quantifications for the missing values of a given datasets based on its existing observations and the observations from other datasets.

# Implementation in R

All code and data is available in the *Matrix Completion in R* folder.

The R script `main_function.R` contains the main functions of the algorithm, along with the description of the arguments.

The csv file `InfluenzaData.csv` contains six real-world influenza datasets (Fonville 2014, Tables S1, S3, S5, S6, S13 and S14) analyzed in the paper (Tal Einav and Rong).

The R script `example_RF.R` contains some example R codes that 

(i) predict the antibody responses against the virus "A.PANAMA.2007.99" in a vaccination study (Fonville 2014, Table 14) using the data from another vaccination study (Fonville 2014, Table S13), and 

(ii) predict the antibody responses against the virus "A.AUCKLAND.5.96" in a vaccination study (Fonville 2014 Table 14) using the data from other three vaccination studies (Fonvile 2014, Tables S5, S6 and S13). 

## Getting Started

To apply our method to an example dataset, follow the three steps below.

1. Download the dataset `InfluenzaData.csv`.
2. Load the R packages and R functions in `main_function.R`.
3. Run the R script `example_RF.R`.


# Implementation in Mathematica

All code and data is available in the *Matrix Completion in Mathematica* folder. The Mathematica notebook `Matrix Completion.nb` contains detailed descriptions of the datasets and matrix completion algorithm, and it provides all the code necessary to recreate the plots in this work.

With the Initialization section loaded, matrix completion of a virus-of-interest in a table-of-interest proceeds as:
 ###
	completionPredictions[{virusOfInterest, tableOfInterest}]
###

## Contact
For further questions and inquiries, please contact Tal Einav (tal.einav@lji.org) or Rong Ma (rongm@stanford.edu).
