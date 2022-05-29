# Cross-Study Matrix Completion via Random Forest

A random forest algorithm for matrix completion across multiple studies.

For a collection of datasets generated from multiple studies, where the measured features may have little-to-no overlap across the studies, our method provides predictions along with uncertainty quantifications for the missing values of a given datasets based on its existing observations and the observations from other datasets.

Our method takes as input the target data matrix whose features (columns) are to be completed, and one or a few additional datasets containing (partial) measurements of the feature-of-interest. The method would output the predicted values for the feature-of-interest, along with their predicted root mean squared error (RMSE).


# Content

The R script `main_function.R` contains the main functions of the algorithm, along with the description of the arguments.

The csv file `InfluenzaData.csv` contains six real-world influenza datasets (Fonville 2014, Tables S1, S3, S5, S6, S13 and S14) analyzed in the paper (Tal Einav and Rong).

The R script `example.R` contains some example R codes that 
(i) predict the antibody responses against the virus "A.PANAMA.2007.99" in a vaccination study (Fonville 2014, Table 14) using the data from another vaccination study (Fonville 2014, Table S13), and 
(ii) predict the antibody responses against the virus "A.AUCKLAND.5.96" in a vaccination study (Fonville 2014 Table 14) using the data from other three vaccination studies (Fonvile 2014, Tables S5, S6 and S13). 

# Get Started

To apply our method to an example dataset, follow the three steps below.

1. Download the dataset `InfluenzaData.csv` from the current directory.
2. Load the R packages and R functions in `main_function.R`.
3. Run the R script `example.R` in the current directory.

For further questions and inquiries, please contact Rong Ma (rongm@stanford.edu).
