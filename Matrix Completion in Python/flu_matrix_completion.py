import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from scipy.odr import *
import os
import pickle
from sklearn.metrics import mean_squared_error

# Might wanna remove this if things get hairy
import warnings
warnings.filterwarnings('ignore')

def train_tree(target_table: "pd.DataFrame",
               source_tables: "list(pd.DataFrame)",
               feature_t: "str",
               selected_viruses_list: "bool OR list(str)" = False,
               n_feature = 5,
               f_sample=0.3,
               k=1,
               comparison_name=None):
    '''
    Function performs:
    - Data cleaning to ensure that viruses have sufficient sera data
    - Selection of virus features which are found in training dataset
    - Virus Selection: selected_viruses_list can be either False for random virus selection or a list of viruses

    Output:
    - For a single target virus, output a list containing:
        - dtr
        - RMSE
        - cross_RMSE
        - virus_col_sel
        - feature_t
        
    Assumptions:
    - This function should be interchangeable with an ensemble tree trainer
    - Target virus which the tree has been trained on is found in training data
        - Cannot train a tree to predict a virus otherwise!
    - Each source table is used for training a tree and finding its
      associated cross-validation RMSE. Table which you want to make
      predictions on is also required so that trees can be trained 
      with relevant viruses.
    - target_table should be a dataframe where columns are viruses
      and row indices are seras. The value for each virus-serum pair
      is an HAI score for binding.
    - tables source_tables are analagous to target_table in structure
    - When talking about cross-validation RMSE,
      I train a tree on data (5 selected viruses X subset of sera) and make predictions
      on testing/validating data (5 selected viruses X all viable sera that weren't used for training).
      I believe that I am cross validating on Sera as selected viruses remain constant
      accuracy of a tree trained on
    '''
#     np.random.seed(42) # Random seed selection # Ummm, setting random seed will make all trees of a target train on the same viruses (really bad lol)

    '''
    Data Preprocessing:
    
    (1) Renaming testing data and training data: data_t, data_assist_train
    (2) Create list of virus names which are covered by >80% of sera in training data, excluding target feature virus: f_col_ind_train
        - To be used for selecting usable features in training data
    (3*) Create list of virus names which intersect training data and f_col_ind_train (viruses covered by >80% of sera): f_tmp_ind
        - Might be redundant
    (3) Create list of virus names which have more than 2 sera values in testing data: f_feasible
        - Honestly sounds a little arbitrary, but it is in Rong's R code 
    (4) Create list of virus names which are found in both f_col_ind_train and testing data: f_feasible
        - To be used for selecting usable features in both training and testing data that we can feasibly test 
            - Only these intersecting featuers will work for: 
                1.) finding cross validation RMSE (RMSE)
                2.) evaluating tree performance on testing data i.e. finding cross table RMSE (cross_RMSE)
    (5) Drop all sera (rows) which have NAN values for the target feature virus (feature_t): data_assist_train
        - This is the only real change to any data table so far (training data)
    '''
#     print(comparison_name)
    tree_dict = dict() # Will contain trained tree, cross-validation RMSE, and (truth, predictions) for cross validation 
    for idx, train_table in enumerate(source_tables):
        # Renaming testing data (data_t) and training data (data_assist_train)
        data_t = target_table # (1) Data that we want to make prediction on 
        data_assist_train = train_table # (1) Main dataframe used in training and validation (Perform log10 transform)
        # Choose viruses that are covered by 80% of sera 
        if (data_assist_train.apply(lambda x: x.count(), axis=0) / data_assist_train.shape[0] > 0.8).sum() > n_feature:
            # f_col_ind_train should be a list of viruses covered by >80% of sera
            
            f_col_ind_train = list(data_assist_train.columns[(data_assist_train.apply(lambda x: x.count(), axis=0) / data_assist_train.shape[0] > 0.8).tolist()]) # (2) Only consider viruses that are covered by > 80% of sera in study
            if feature_t in f_col_ind_train: # Exclude target feature virus
                f_col_ind_train.remove(feature_t) # (2)
        else:
            print(f"n_feature too large for assisting data {k}! Skipped to next data.")
#         print(f'f_col_ind_train: {f_col_ind_train}')
        # WARNING: .get_loc method will return a list of bools instead of the expected index integer value if multiple instaces of col exist in data_t.columns
        f_tmp_ind = [data_t.columns.get_loc(col) for col in f_col_ind_train if col in data_t.columns] # (3*) Find indices of columns that intersect f_col_ind_train and data_t
        # Check f_tmp_ind
#         print(f'f_tmp_ind: {f_tmp_ind}')
        f_feasible = [col for col in data_t.columns[f_tmp_ind] if data_t[col].count() > 2] # (3) Choose viruses that have more than 2 sera values in prediction data
        f_col_ind_train = [col for col in f_col_ind_train if col in f_feasible] # (4) Intersects cols from f_feasible and f_col_ind_train
        if len(f_feasible) < 2:
            print(f"n_feature too large for assisting data {k}! Skipped to next data.")
            return
        data_assist_train = data_assist_train.dropna(subset=[feature_t]) # (5) Drops NAs in training data for target

        '''
        Training:
        
        Training conditions: Perform checks on validity of training data
        (1) Training dataset must have sera binding data on target virus
        (2) Target feature is in training data (only way target dataset error (cross_RMSE) can be computed)
        '''
        if data_assist_train.shape[0] == 0: # (1)
            return
        if feature_t in data_t:
            if sum(np.isnan(data_t[feature_t])) == len(data_t[feature_t]): # (2)
    #                print(f"sum(np.isnan(data_t[feature_t])): {sum(np.isnan(data_t[feature_t]))}")
                return
        else:
    #         print(f"{feature_t} not in data_t")
            return
        # Training step (and single cross validation):
        if selected_viruses_list: # If selected_viruses_list is not False, I'm assuming its just a list of virus names
            virus_col_sel = np.array(selected_viruses_list) # Viruses we specifically want to train on and use for predictions in predicting data
        else:
            features_list = list(data_assist_train.columns) # Make sure feature_t isn't in features to train on
            features_list.remove(feature_t)
            virus_col_sel = np.random.choice(f_feasible, n_feature, replace=True)

        sera_row_sel = np.random.choice(data_assist_train.shape[0], int(data_assist_train.shape[0] * f_sample), replace=True) # Randomly selected sera
        # We want to include the column of f_t_ind into our data_train
        data_train = data_assist_train.iloc[sera_row_sel][np.append(virus_col_sel, feature_t)] # Training data subset covering virus_col_sel viruses and sera_row_sel sera
        col_mean_t = data_train.apply(lambda x: x.mean(), axis=1) # Prepare to mean center the training data
        data_train = data_train - np.outer(np.ones(data_train.shape[1]), col_mean_t).T # Mean center the training data
        data_train.columns = np.append(virus_col_sel, "target")
        dtr = DecisionTreeRegressor(min_samples_split=5)
        # print(data_train)
        dtr.fit(data_train.iloc[:, :-1], data_train["target"]) # Train on selected viruses and sera, evaluate with target virus
        # Compile validation dataset, really performing a single cross validation (evaluate on data complementary to training data)
        data_test = data_assist_train.drop([data_assist_train.index[idx] for idx in sera_row_sel], axis=0)[np.append(virus_col_sel, feature_t)] # Testing data subset covering virus_col_sel viruses and all sera not covered by training data 
        col_mean_t = data_test.apply(lambda x: x.mean(), axis=1)
        data_test = data_test - np.outer(np.ones(data_test.shape[1]), col_mean_t).T
        pred_t = dtr.predict(data_test.iloc[:, :-1]) # Make prediction on target using unforeseen data
        RMSE = np.sqrt(np.mean((pred_t - data_test[feature_t]) ** 2))

        # Compile testing dataset using data_t for cross-table RMSC
        cross_virus_col_sel = virus_col_sel # Viruses we specifically want to train on and use for predictions in predicting data
        cross_sera_row_sel = np.random.choice(data_t.shape[0], int(data_t.shape[0] * f_sample), replace=True) # Randomly selected sera        data_test = data_assist_train.drop([data_assist_train.index[idx] for idx in sera_row_sel], axis=0)[np.append(virus_col_sel, feature_t)] # Testing data subset covering virus_col_sel viruses and all sera not covered by training data 
        cross_data_test = data_t[np.append(cross_virus_col_sel, feature_t)] # Testing data subset covering virus_col_sel viruses and all sera not covered by training data 
        cross_col_mean_t = cross_data_test.apply(lambda x: x.mean(), axis=1)
        cross_data_test = cross_data_test - np.outer(np.ones(cross_data_test.shape[1]), cross_col_mean_t).T
        cross_pred_t = dtr.predict(cross_data_test.iloc[:, :-1]) # Make prediction on target using unforeseen data

        # OKOKOK, we can only find cross_RMSE if target table actually contains target feature...
        cross_RMSE = np.sqrt(np.mean((cross_pred_t - cross_data_test[feature_t]) ** 2)) 
        if np.isnan(cross_RMSE):
            '''
            Problems with cross_data_test[feature_t]
            - Select better viruses to train on! Likely reason for failure is that im not choosing trainable viruses
            '''
#             print("Problem with computing cross_RMSE!!!")
#             print(cross_virus_col_sel)
#             print(cross_data_test)
#             print("PREDICTIONS:")
#             print(cross_pred_t)
#             print()
    return [dtr, RMSE, cross_RMSE, virus_col_sel, feature_t, cross_pred_t, comparison_name]
    
def m_best_trees_trainer(target_table: "pd.DataFrame",
                         source_tables: "list(pd.DataFrame)",
                         feature_t: "str",
                         selected_viruses_list: "bool OR list(str)" = False,
                         n_feature = 5,
                         f_sample=0.3,
                         train_trees=10,
                         best_trees=5,
                         k=1,
                         comparison_name=None):
    '''
    This function will call the subroutine N number of times to generate N trees, from which the M best may be selected/returned
    Might also want to store a list of the target + training viruses for each tree made (might be like a bar code or signature)
    Once a bunch of trees are returned for the comparison between the target and training datasets (including a map from "tree name" : "list of involved viruses")
    we may compute an error line for each training virus in the comparison plot. With this Error line, we have our means of computing the worst case error of a 
    prediction.
    '''
    train_tree_list_list = [] # A single train_tree_list_list corresponds to the current target virus
    for i in range(train_trees): # Make some trees
        train_tree_list = train_tree(target_table = target_table,
                                     source_tables = source_tables,
                                     feature_t = feature_t,
                                     selected_viruses_list = selected_viruses_list,
                                     n_feature = n_feature,
                                     f_sample = f_sample,
                                     k = k,
                                     comparison_name=comparison_name)
        if train_tree_list is None:
#             print("NONE???")
            return train_tree_list
        dtr, RMSE, cross_RMSE, virus_col_sel, feature_t, cross_pred_t, comparison_name = train_tree_list
        train_tree_list_list.append(train_tree_list)
    # Rank trees in descending order by cross validation RMSE (training error, not cross table RMSE which iis cross_RMSE)
    train_tree_list_list = sorted(train_tree_list_list, key = lambda x:x[1]) 
    return train_tree_list_list[:best_trees]
    
'''
For following functions, read Box 1 and Box 2 from Tal's paper

pseudocode:
for each comparison (other dataset is D_j):
    for each virus (V_j_i):
        1.)
            - select all virus points within current comparison which (in order to prevent "double dipping"):
                (1) Aren't the current target virus
                (2) Don't use any of the viruses used to train the tree for the current target virus
            - With selected points, train a ODR line for current target virus
            - SIGMA_j: Assign virus name an ODR line using a dict?
        2.)
            - MU_j: Find Bayesian Average of predicted value (relies on SIGMA_j)
    - Assign a comparison a virus2ODR dict
'''
def odr_model(params, x):
    a, b = params
    return a * x + b


def find_odr_fitted_params(x_data, y_data, odr_model):
    # Find ODR or perpendicularly fit line on current set of points
    data = RealData(x_data, y_data)
    model = Model(odr_model)
    initial_guess = [1.0, 1.0]  # Initial guess for the parameters
    odr = ODR(data, model, beta0=initial_guess)
    result = odr.run()
    fitted_params = result.beta # Slope and bias parameters for ODR
    return fitted_params

def worst_case_error(RMSE, odr_model, fitted_params, use_MSE=False):
    '''
    Function takes in RMSE, odr model, and trained/fitted parameters.
    Function then returns the max between the RMSE and the error predicted
    from inputting RMSE into the trained odr model.
    
    Assume that RMSE is a list or array
    '''
    # print(len(RMSE))
    
    output = []
    output_w_MSE = []
    odr_values = []
    if len(RMSE) == 0:
        return output
    for idx in range(len(RMSE)):
        odr_value = odr_model(fitted_params, RMSE[idx])
        odr_values.append(odr_value)
        output.append(max(RMSE[idx], odr_value))
    f_RMSE = mean_squared_error(odr_values, RMSE)**0.5
    if use_MSE:
        for idx in range(len(RMSE)):
            odr_value = odr_model(fitted_params, RMSE[idx])
            odr_values.append(odr_value)
            output_w_MSE.append(max(RMSE[idx], odr_value+f_RMSE))
        return output_w_MSE
    return output

def convert_raw_dtr_predictions(data, dtr_list):
    '''
    Input for trained decision trees must be log10 transformed
    and mean centered HAI values corresponding to the {n_feautes}
    virus featues which the decision tree was trained on. 
    
    Data should be the table which a prediction will be made from.
    
    Output will be HAI values that will essentially be mean_centered
    and log10 transformed. Due to this, in order to retrieve actual
    HAI values, these transformations must be reversed.
    '''
    dtr = dtr_list[0]
    col_select = dtr_list[3].tolist() # np.random.choice(df.columns, 5)
    target = dtr_list[4]
    col_select.append(target)
    data_select_target = data[col_select] # Might not need to drop NANs
    data_select = data_select_target.iloc[:,:-1]
#     print(data_select)
    col_mean_t = data_select.apply(lambda x: x.mean(), axis=1)
    data_select = data_select - np.outer(np.ones(data_select.shape[1]), col_mean_t).T
    pred_t = dtr.predict(data_select.to_numpy()) # Make prediction on target using unforeseen data
    pred_t_uncentered = pred_t + col_mean_t

    HAI_predictions = 10**(pred_t_uncentered)
    HAI_measurements = 10**data_select_target[target]
#     print(HAI_measurements)
    return HAI_predictions.to_numpy().reshape(1, -1), HAI_measurements.to_numpy().reshape(1, -1), pred_t

def average_convert_raw_dtr_predictions(data, dtr_lists_list):
    HAI_predictions_list = []
    for dtr_list in dtr_lists_list:
        HAI_predictions, HAI_measurements, raw_predictions = convert_raw_dtr_predictions(data=data, dtr_list=dtr_list)
        HAI_predictions_list.append(HAI_predictions)
    
    HAI_predictions_df = pd.DataFrame()
    for idx, i in enumerate(HAI_predictions_list):
        HAI_predictions_df[idx] = i.tolist()[0]
    mu = HAI_predictions_df.mean(axis=1)
    return mu, HAI_measurements, HAI_predictions_df, raw_predictions

def combine_predictions(target_virus_name,
                        target_table_name,
                        source_table_names: "list",
                        comparison_combiner_dict):
    '''
    Uses formula from Box 1, section 3
    '''
    predictions_raw = []
    errors_raw = []
    for j, source_table_name in enumerate(source_table_names):
        comparison_name = f"{source_table_name} TO {target_table_name}"
        mu_j = comparison_combiner_dict[comparison_name][target_virus_name]['mu'] # Mean of HAI predictions for virus
        sigma_j, error_actual = comparison_combiner_dict[comparison_name][target_virus_name]['error_predict_actual'] # Only need sigma_j (error_predict)
        sigma_j = sigma_j[0]
        pred_j = (mu_j/(sigma_j**2)) / (1/(sigma_j**2))
        error_j = 1 / (sigma_j**2)
        predictions_raw.append(pred_j)
        errors_raw.append(error_j)
    predictions_final = sum(predictions_raw)
    errors_final = 1 / np.sqrt(sum(errors_raw))
    return predictions_final, errors_final

class HI_data_tables():
    '''
    USE:
    This class is to be used by the matrix_completion class for training
    matrix completion models for the purpose of imputing HI scores for antisera
    entries with missing but viable antisera-virus interactions
        - Viable antisera-virus interactions are those which have previously been
          seen between the target virus and other antisera with sufficient
          overlapping interactions with the antisera given as input to the model

    CLASS CONTENTS:
    (1) A pandas dataframe where rows are antisera (index) and columns are
        viruses. The values of the dataframe are log10 transformed HI scores. 
        - Datasets of the same form (antisera rows, virus columns) can be added

    (2) A pandas dataframe corresponding to each antisera. This dataframe will
        contain information such as groupID or metadata to enable the grouping
        or identification of antisera
        - ex. Fonville data has 6 groups (TableS1, TableS3, ..., TableS14),
          the groupID column of dataframe (2) will specify the group which an
          antisera belongs to.

    (3) A pandas dataframe corresponding to each virus. This dataframe will
        contain information such as antigenic cluster or metadata to enable the
        grouping or identification of virus.
    '''
    def __init__(self):
        self.HI_data = pd.DataFrame(columns=["sampleID"]).set_index("sampleID")
        self.antisera_table = pd.DataFrame(columns=["sampleID", "groupID"]).set_index("sampleID")
        self.virus_table = pd.DataFrame() # For now, this is unused

    def add_HI_data(self, HI_df:pd.DataFrame, antisera_df:pd.DataFrame, virus_df=None):
        '''
        Inputs:

        HI_df: Dataframe that where indices are antisera names and columns are
        viruses names. Values must be log10 transformed HI values.

        antisera_df: Dataframe where indices are antisera names and values are
        the group name corresponding to each antisera. Default column
        for this dataframe is sampleID which can be used to group antisera.

        virus_df: Dataframe where indices are virus names. For now this isn't
        used, but may see use if information about antigenic clustering is needed
        '''
        # Concat self.HI_data with HI_df given as input
        HI_df = HI_df.rename_axis('sampleID') # Remame index col name to match HI_data index col name
        self.HI_data = pd.concat([self.HI_data, HI_df], axis=0) # Will overwrite overlapping antisera entries if given any
        # Concat self.antisera_table with antisera_df given as input 
        antisera_df = antisera_df.rename_axis('sampleID') # Remame index col name to match antisera_table index col name
        self.antisera_table = pd.concat([self.antisera_table, antisera_df], axis=0)
        # Nothing done with self.virus_table yet
        return
    
    def select_HI_data_by_group(self, group:"list[str]", sort_by_year=True):
        '''
        Given name of data group, return corresponding HI_data entries as dataframe
        '''
        indices = [i in group for i in self.antisera_table['groupID'].to_numpy()]
        selected_df = self.HI_data.iloc[indices]
        if sort_by_year:
            virusID_sorted_by_year = self.virus_table.sort_values(by='Year').virusID.tolist()
            sorted_HI_cols = [i for i in virusID_sorted_by_year if i in selected_df.columns]
            selected_df = selected_df[sorted_HI_cols]
        return selected_df
    
    def compute_virus_dates(self,):
        self.virus_table['virusID'] = list(self.HI_data.columns)
        # Extract the years using a regular expression
        self.virus_table['Year'] = [i.split('/')[-1] for i in self.virus_table['virusID']]
        self.virus_table['Year'] = self.virus_table['Year'].astype(str).str.extract(r'(\d{2,4})')

        # Convert two-digit years to four-digit years   
        self.virus_table['Year'] = pd.to_numeric(self.virus_table['Year'], errors='coerce')
        self.virus_table['Year'] = self.virus_table['Year'].apply(lambda x: x + 1900 if (x <= 99 and x >= 68) else (x + 2000 if x <= 68 else x))
        return
    

    def data_counts(self):
        return self.antisera_table.groupby('groupID').size().reset_index(name='Count')



class transferability_comparisons():
    '''
    This class is used to perform intra and inter RMSE analyses given a list of
    datasets where each dataset corresponds to its own group.

    Use case with Tal's' paper:
    Given the fonville data which was grouped into 6 tables, we can use this class
    to train a set of decision trees using training data for each table to
    target viruses in each of the other 5 tables.
    '''

    def __init__(self, HI_data_tables):
        self.HI_data_tables = HI_data_tables
        self.comparison_dict = None
        self.intra_RMSE_dict = None
        self.cross_RMSE_dict = None
        self.comparison_virus_ODR_df_dict = None
        self.comparison_combiner_dict = None

    def save_data(self, save_to):
        with open(save_to, 'wb') as file:
            data = [self.comparison_dict,
                    self.intra_RMSE_dict,
                    self.cross_RMSE_dict,
                    self.comparison_virus_ODR_df_dict,
                    self.comparison_combiner_dict]
            pickle.dump(data, file)

    def load_data(self, path):
        with open(path, 'rb') as file:
            self.comparison_dict, self.intra_RMSE_dict, self.cross_RMSE_dict, self.comparison_virus_ODR_df_dict, self.comparison_combiner_dict = pickle.load(file)
            # self.comparison_dict, self.intra_RMSE_dict, self.cross_RMSE_dict, self.comparison_virus_ODR_df_dict = pickle.load(file)


    def train_comparison_trees(self, train_trees=1, best_trees=1, verbose=False):
        
        table_names = list(self.HI_data_tables.antisera_table['groupID'].unique())
        table_list = [self.HI_data_tables.select_HI_data_by_group(group=i, sort_by_year=True) for i in table_names]
        comparison_dict = dict()
        for idx, source_table in enumerate(table_list):
            source_table_name = table_names[idx]
            for jdx, target_table in enumerate(table_list):
                if idx != jdx: # Ensure that target table is not the same as source table
                    # Dict names
                    train_tree_list = None
                    target_table_name = table_names[jdx]
                    comparison_name = f"{source_table_name} TO {target_table_name}"
                    tree_dict = dict() # Contains {trees and RMSE data} collected from training a tree on target features of target_table
                    # Preparing tree parameters
                    feature_targets = list(target_table.columns) # List of viruses to make a tree/prediction on
                    for i, feature_target in enumerate(feature_targets): # Make a tree for each virus in target_table if virus in source_table
                        if feature_target in source_table.columns:
                            train_tree_list = m_best_trees_trainer(target_table = target_table,
                                                                source_tables = [source_table],
                                                                feature_t = feature_target,
                                                                selected_viruses_list = False,
                                                                n_feature = 5,
                                                                f_sample=0.3,
                                                                train_trees=train_trees,
                                                                best_trees=best_trees,
                                                                k=1,
                                                                comparison_name=comparison_name)
                        else:
                            print(f"{table_names[idx]} does not contain {feature_target} which {table_names[jdx]} seeks!")
                        if train_tree_list is not None and len(train_tree_list) != 0:
                            tree_dict[feature_target] = train_tree_list
                    comparison_dict[comparison_name] = tree_dict
        #             print("----------------")
                    if verbose:
                        print(f"Comparison from {source_table_name} to {target_table_name} completed")
        self.comparison_dict = comparison_dict

        # Compute intra and cross RMSEs for each tree
        intra_RMSE_dict = dict()
        cross_RMSE_dict = dict()
        for idx, source_table_key in enumerate(table_names):
            source_table_name = source_table_key
            for jdx, target_table_key in enumerate(table_names):
                if idx != jdx: # Ensure that target table is not the same as source table
                    # Dict names

                    target_table_name = target_table_key
                    comparison_name = f"{source_table_name} TO {target_table_name}"
                    intra_RMSE_data = []
                    cross_RMSE_data = []
                    for target in list(comparison_dict[comparison_name].keys()):
                        tree_data = comparison_dict[comparison_name][target] # Assuming that only one source table was used in training tree
                        for l in tree_data:
                            intra_RMSE_data.append(l[1])
                            cross_RMSE_data.append(l[2])
                    intra_RMSE_dict[comparison_name] = intra_RMSE_data
                    cross_RMSE_dict[comparison_name] = cross_RMSE_data
            self.intra_RMSE_dict = intra_RMSE_dict
            self.cross_RMSE_dict = cross_RMSE_dict
        return
    
    def plot_comparisons(self, save_to=None, **kwargs):
        table_names = list(self.HI_data_tables.antisera_table['groupID'].unique())
        table_list = [self.HI_data_tables.select_HI_data_by_group(group=i, sort_by_year=True) for i in table_names]
        nrows, ncols = len(table_names), len(table_names)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(100, 100))

        for idx, source_table in enumerate(table_list):
            source_table_name = table_names[idx]
            for jdx, target_table in enumerate(table_list):
                if idx != jdx: # Ensure that target table is not the same as source table
                    # Dict names
                    target_table_name = table_names[jdx]
                    comparison_name = f"{source_table_name} TO {target_table_name}"
                    
                    x = self.intra_RMSE_dict[comparison_name]
                    y = self.cross_RMSE_dict[comparison_name]

                    axs[idx, jdx].scatter(x,y, **kwargs)
                    axs[idx, jdx].set_title(comparison_name, fontsize = 30)
                    axs[0,0].set_ylabel("Cross Table RMSE", fontsize = 30)
                    axs[idx, jdx].tick_params(axis='both', which='major', labelsize=20)
                    axs[idx, jdx].grid()
                    axs[idx, jdx].plot([0, 1], [0, 1], transform=axs[idx, jdx].transAxes)
                    axs[idx, jdx].set_xlim([0, 1])
                    axs[idx, jdx].set_ylim([0, 1])
                fig.suptitle("Cross VS Intra Table RMSE", fontsize=30)
                fig.supxlabel('Intra Table RMSE', fontsize=30)
        if save_to is not None:
            plt.savefig(save_to)
        fig.show()
        return
    
    def compute_comparison_virus_ODR_df_dict(self, verbose=False):
        '''
        Method computes comparison_ODR_df_dict and comparison_virus_ODR_df_dict for object.
        These dicts contain the (1) per dataset per virus and (2) per dataset
        worst case errors respectively.
        '''
        comparison_ODR_df_dict = dict() # Will contain ODR model per comparison done (not virus specific, general estimation of error per comparison)
        comparison_virus_ODR_df_dict = dict() # Will contain all ODR models
        for comparison_name in self.comparison_dict.keys():
            if verbose:
                print(f"{comparison_name}: Extracting ODR datapoints and training ODR per virus")
            self.comparison_dict[comparison_name] 
            virus_ODR_df_dict = dict()
            for v_j_i in self.comparison_dict[comparison_name].keys():
                sigma_training_list = []
                sigma_actual_list = []
                unacceptable_features = [] # If tree was trained on any of these features, don't consider it for ODR
                df = pd.DataFrame()
                for idx, v_j_i_tree in enumerate(self.comparison_dict[comparison_name][v_j_i]): # Iterate over trained tree lists given by train_tree function
                    training_features = v_j_i_tree[3]
                    target_feature = v_j_i_tree[4]
            #         unacceptable_features.extend(training_features) # Not sure if this counts as double dipping
                    unacceptable_features.append(target_feature)

                for v_j_i_inner in self.comparison_dict[comparison_name].keys(): # Iterate over all other viruses
                    if v_j_i != v_j_i_inner: # Only consider viruses which aren't the target for finding ODR
                        for idx_inner, v_j_i_tree_inner in enumerate(self.comparison_dict[comparison_name][v_j_i_inner]): # Iterate over trained tree lists given by train_tree function
                            sigma_training = v_j_i_tree_inner[1]
                            sigma_actual = v_j_i_tree_inner[2]
                            training_features = v_j_i_tree_inner[3]
                            target_feature = v_j_i_tree_inner[4]
            #                 print(f"target_feature not in unacceptable_feature: {target_feature not in unacceptable_features}")
            #                 print(f"not any([feature in unacceptable_features for feature in training_features]): {not any([feature in unacceptable_features for feature in training_features])}")
                            if target_feature not in unacceptable_features and not any([feature in unacceptable_features for feature in training_features]):
                                sigma_training_list.append(sigma_training)
                                sigma_actual_list.append(sigma_actual)

                # Find ODR or perpendicularly fit line on current set of points
                fitted_params = find_odr_fitted_params(x_data=sigma_training_list, 
                                                    y_data=sigma_actual_list,
                                                    odr_model=odr_model)
                # Construct dataframe with sigma_training, sigma_actual, and worst case error predicted from ODR
                df['sigma_training'] = sigma_training_list
                df['sigma_actual'] = sigma_actual_list
                df['worst_case_error'] = worst_case_error(sigma_training_list, odr_model, fitted_params)
                virus_ODR_df_dict[v_j_i] = (df, odr_model, fitted_params) # Might not need odr_model function as an arg if I make it a global function
            comparison_virus_ODR_df_dict[comparison_name] = virus_ODR_df_dict
            
            x_data = self.intra_RMSE_dict[comparison_name]
            y_data = self.cross_RMSE_dict[comparison_name]
            fitted_params = find_odr_fitted_params(x_data=x_data, 
                                                y_data=y_data,
                                                odr_model=odr_model)
            comparison_ODR_df_dict[comparison_name] = (x_data, y_data, odr_model, fitted_params)

            
        self.comparison_ODR_df_dict = comparison_ODR_df_dict
        self.comparison_virus_ODR_df_dict = comparison_virus_ODR_df_dict

    def compute_comparison_combiner_dict(self):
        '''
        NOTES:
        - Might want to ensure all dfs contain sera names as indices!
            - Acutally, assuming that the target table determines the number of predictions
            made, indexing might not be necessary!
            - e.g. We have 2 comparisons "tableS1 TO tableS6" and "tableS5 TO tableS6",
            both will end up having 160 predictions each since there are 160 predictable viruses
            in tableS6
            - Still might wanna index for clarity's sake
        '''

        comparison_combiner_dict = dict() # This dict is needed for combining Box 1 part 3

        for comparison_idx, comparison_name in enumerate(list(self.comparison_dict.keys())): # Choose comparison
            comparison_combiner_dict[comparison_name] = dict()
            source_table, _, target_table = comparison_name.split(" ") # Find source and target tables of comparison
            for virus_idx, virus_name in enumerate(list(self.comparison_dict[comparison_name].keys())): # Choose target virus in comparison
                data = self.HI_data_tables.select_HI_data_by_group(group=target_table)
                # Retrieve decision tree, ODR model, and predictions
                dtr_lists_list = self.comparison_dict[comparison_name][virus_name] # Retrieve decision tree data
                df, odr_model, fitted_params = self.comparison_virus_ODR_df_dict[comparison_name][virus_name] # Retrieve ODR
                mu, HAI_measurements, HAI_predictions_df, raw_predictions = average_convert_raw_dtr_predictions(data, dtr_lists_list) # Find AVG HAI preds and measurements        
                # Find error for each HAI prediction using ODR
                sigma_training_list = [i[1] for i in dtr_lists_list] # See Box 1 section 2 for reasoning
                mean_sigma_training = sum(sigma_training_list)/len(sigma_training_list) # Take mean
                error_predict = worst_case_error(RMSE = [mean_sigma_training], odr_model = odr_model, fitted_params = fitted_params) # Find worst case error
                error_actual = 10**np.sqrt(np.nanmean((np.log10(HAI_measurements.flatten()) - np.log10(mu.to_numpy()))**2))
                # Error bar data (Warning: Only valid for viruses with HAI measurements)
                x = np.log10(HAI_measurements.tolist()[0])
                y = np.log10(mu.tolist())
                errors = error_predict * len(x)
                # Note that viruses without HAI measurements to begin with won't have valid error bar data
                comparison_combiner_dict[comparison_name][virus_name] = {
                    'mu': mu,
                    'error_predict_actual': (error_predict, error_actual),
                    'error_bar_data': (x, y, errors)
                }
        self.comparison_combiner_dict = comparison_combiner_dict

    def plot_comparisons_with_ODRs(self, save_to=None, **kwargs):
        table_names = list(self.HI_data_tables.antisera_table['groupID'].unique())
        table_list = [self.HI_data_tables.select_HI_data_by_group(group=i, sort_by_year=True) for i in table_names]
        nrows, ncols = len(table_names), len(table_names)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(100, 100))

        for idx, source_table in enumerate(table_list):
            source_table_name = table_names[idx]
            for jdx, target_table in enumerate(table_list):
                if idx != jdx: # Ensure that target table is not the same as source table
                    # Dict names
                    target_table_name = table_names[jdx]
                    comparison_name = f"{source_table_name} TO {target_table_name}"
                    # Scatter points
                    x = self.intra_RMSE_dict[comparison_name]
                    y = self.cross_RMSE_dict[comparison_name]
                    # Worst case error computed from ODR trained on comparison
                    fitted_params = self.comparison_ODR_df_dict[comparison_name][-1]
                    odr_model = self.comparison_ODR_df_dict[comparison_name][-2]
                    x_values = np.linspace(0, 1, 100)
                    y_values = worst_case_error(x_values, odr_model, fitted_params)
                    axs[idx, jdx].plot(x_values, y_values, label=f"ODR Model", color='red')
                    
                    axs[idx, jdx].scatter(x,y,  **kwargs)
                    axs[idx, jdx].set_title(comparison_name, fontsize = 30)
                    axs[0,0].set_ylabel("Cross Table RMSE", fontsize = 30)
                    axs[idx, jdx].tick_params(axis='both', which='major', labelsize=20)
                    axs[idx, jdx].grid()
                    axs[idx, jdx].plot([0, 1], [0, 1], transform=axs[idx, jdx].transAxes, 
                                       label="Diagonal Line (y = x)", linestyle='dashed',
                                       color='green')
                    slope = fitted_params[0]
                    transferability = 1/slope
                    axs[idx, jdx].text(0.95, 0.05, f'{transferability:.2f}',
                                       fontsize=30, color='red', ha='right', va='bottom')
                    axs[idx, jdx].set_xlim([0, 1])
                    axs[idx, jdx].set_ylim([0, 1])
                fig.suptitle("Cross VS Intra Table RMSE", fontsize=30)
                fig.supxlabel('Intra Table RMSE', fontsize=30)
        if save_to is not None:
            plt.savefig(save_to)
        fig.show()
        return
    
    def plot_ODR(self, comparison_name, sample_name, train_trees, best_trees):
        '''
        Function plots out ODR line trained on datapoints relevant to a target virus's tree RMSE values
        '''
        ODR_df_dict = self.comparison_virus_ODR_df_dict[comparison_name][sample_name]
        odr_model = ODR_df_dict[1]
        fitted_params = ODR_df_dict[-1]
        # Use ODR model to retrieve worst case errors
        x_values = np.linspace(0, 1, 100)
        y_values = worst_case_error(x_values, odr_model, fitted_params)
        # Retrieve simga_training and sigma_acutal relevant to virus
        x_data = ODR_df_dict[0]['sigma_training']
        y_data =  ODR_df_dict[0]['sigma_actual']
        print(y_values)
        print()
        # Plot data points
        sns.scatterplot(x=x_data, y=y_data, label="Data")
        # Add the ODR model line to the plot
        sns.lineplot(x=x_values, y=y_values, label=f"ODR Model", color='red')
        
        # Set labels, title, and legend
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"ODR Model Fitted Line: Train {train_trees}, Best {best_trees} trees;\n{comparison_name} Sample {sample_name}")
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        sns.lineplot(x=[0, 1], y=[0, 1], label="Diagonal Line (y = x)", linestyle='--', color='green')

        # Display the plot
        plt.savefig("figs/"+ f"ODR Model: Train {train_trees}, Best {best_trees} trees: {comparison_name} Sample {sample_name.replace('/', '_')}")
        plt.show()
        return