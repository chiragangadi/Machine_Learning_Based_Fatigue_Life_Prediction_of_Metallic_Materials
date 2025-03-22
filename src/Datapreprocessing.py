import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import TensorDataset, DataLoader
from .CustomFunctions import custom_data_split_quartile, custom_data_split_categorical, quartile_based_kfold, handle_outliers_iqr, handle_outliers_zscore, pairplots, contourplots, categorical_based_kfold

class DataCleaning(BaseEstimator, TransformerMixin):
    """
    A custom data cleaning transformer for preprocessing datasets.

    Parameters:
    -----------
    columns_remove_missingvalues : list
        List of column names where rows with missing values should be removed.
    columns_filter : dict
        Dictionary where keys are column names and values are lists of allowed strings; 
        rows not containing these values will be filtered out.
    exclude_norunnouts : bool
        If True, removes rows where the specified feature is zero.
    feature_name : str
        Column name used to filter out zero values if `exclude_norunnouts` is True.
    condition_outlierdetection : bool
        If True, outlier detection and handling will be performed.
    columns_outlier : list
        List of columns on which outlier detection should be applied.
    detection_method_factor : dict
        Dictionary specifying the outlier detection method ('method') and factor ('factor') 
        for determining outlier thresholds.
    handling_method : str
        Method for handling detected outliers, e.g., 'remove' or 'cap'.

    Methods:
    --------
    fit(df):
        Returns self; no fitting required as this is a stateless transformer.
    
    transform(df):
        Applies the following data cleaning steps:
        - Removes rows with missing values in specified columns.
        - Filters rows based on allowed string values in specified columns.
        - Removes rows where the specified feature is zero if `exclude_norunnouts` is True.
        - Detects and handles outliers in specified columns using IQR or Z-score methods.
        
        Returns the cleaned DataFrame.
    """
    
    def __init__(self, columns_remove_missingvalues, columns_filter, exclude_norunnouts, feature_name, condition_outlierdetection, columns_outlier, detection_method_factor, handling_method):
        self.columns_remove_missingvalues = columns_remove_missingvalues
        self.columns_filter = columns_filter
        self.exclude_norunnouts = exclude_norunnouts
        self.feature_name = feature_name
        self.columns_outlier = columns_outlier
        self.condition_outlierdetection = condition_outlierdetection
        self.detection_method = detection_method_factor['method']
        self.detection_factor = detection_method_factor['factor']
        self.handling_method = handling_method

    def fit(self, df):
        return self
    
    def transform(self, df):
        
        #################################################################################################################################
       
        # Dropping rows containing missing values in specific columns
        df = df.dropna(subset = self.columns_remove_missingvalues).reset_index(drop = True)

        #################################################################################################################################
        
        # Filtering data points with specified strings in list of columns
        for col, strings in self.columns_filter.items():
            indices_list = []
            for i in range(len(df)):
                if df[col][i] not in strings:
                    indices_list.append(i)
            df = df.drop(indices_list).reset_index(drop = True)

        #################################################################################################################################

        if self.exclude_norunnouts == True:
            df = df[df[self.feature_name] != 0].reset_index(drop = True)

        ################################################################################################################################

        if self.condition_outlierdetection == True:
            if self.detection_method == 'iqr':
                df = handle_outliers_iqr(df, self.columns_outlier, self.handling_method, self.detection_factor)
            elif self.detection_method == 'zscore':
                df = handle_outliers_zscore(df, self.columns_outlier, self.handling_method, self.detection_factor)

        ################################################################################################################################

        print('Data cleaning step completed')

        return df


class DataTransformation(BaseEstimator, TransformerMixin):
    """
    A custom transformer for data preprocessing, including feature selection, missing value imputation, 
    categorical encoding, and scaling.

    Parameters:
    -----------
    columns_features : list
        List of feature column names to be used in the transformation.
    column_target : list
        List containing the target column name.
    columns_impute : dict
        Dictionary specifying missing value imputation strategies. 
        Keys are column names, and values are tuples (method, value), where method can be:
        - "number": Replace missing values with a specified number.
        - "string": Replace missing values with a specified string.
        - "mean": Replace missing values with the column mean.
        - "median": Replace missing values with the column median.
    columns_categorical_dataencoding : list
        List of categorical columns to be encoded.
    dataencoding_method : str
        Encoding method for categorical variables. Options:
        - "onehotencoder": Apply one-hot encoding.
        - "ordinalencoder": Apply ordinal encoding.
    con_scale : bool
        If True, applies feature scaling.
    scale_method : str
        Scaling method to use if `con_scale` is True. Options:
        - "normalization": Min-Max scaling.
        - "standardization": Standard scaling.

    Methods:
    --------
    fit(df):
        Returns self; no fitting required as this is a stateless transformer.
    
    transform(df):
        Applies the following data transformation steps:
        - Selects relevant feature and target columns.
        - Imputes missing values based on specified methods.
        - Encodes categorical variables using one-hot or ordinal encoding.
        - Scales numerical features and target variable if `con_scale` is True.

        Returns:
        --------
        - df_normal : DataFrame before scaling.
        - df_scaled : DataFrame after scaling (if scaling is applied).
        - scaler_target : The scaling object applied to the target variable.
        - scale_method : The scaling method used.
    """

    def __init__(self, columns_features, column_target, columns_impute, columns_categorical_dataencoding, dataencoding_method, con_scale, scale_method):
        self.columns_features = columns_features
        self.column_target = column_target
        self.columns_impute = columns_impute
        self.columns_categorical_dataencoding = columns_categorical_dataencoding
        self.dataencoding_method = dataencoding_method
        self.con_scale  = con_scale 
        self.scale_method = scale_method

    def fit(self, df):
        return self

    def transform(self, df):

        df_normal = df

        #################################################################################################################################
        
        # Selecting columns which are features and labels
        df = df[self.columns_features + self.column_target]
       
        #################################################################################################################################

        # Filling missing values in specified columns with different methods of imputation
        for col, method_value in self.columns_impute.items():
            if method_value[0] == "number":
                # Filling missing values with zero in specified columns
                df.loc[:,col] = df[col].fillna(method_value[1])
            elif method_value[0] == "string":
                # Filling missing values with zero in specified columns
                df.loc[:,col] = df[col].fillna(method_value[1])
            elif method_value[0] == "mean":
                # Filling missing values with mean of the column
                df.loc[:,col] = df[col].fillna(df[col].mean())
            elif method_value[0] == "median":
                # Filling missing values with mean of the column
                df.loc[:,col] = df[col].fillna(df[col].median())
            #elif:
                # custom function
            else:
                print(f"Wrong imputer method provided for column: {col}")

        #################################################################################################################################
        
        if self.dataencoding_method == "onehotencoder":
            # Initialize the OneHotEncoder
            encoder = OneHotEncoder()
            # Fit and transform the selected columns
            encoded_columns = encoder.fit_transform(df[self.columns_categorical_dataencoding]).toarray()
            # Create a DataFrame from the encoded columns
            encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(input_features=self.columns_categorical_dataencoding))
            # Concatenate the encoded DataFrame with the original DataFrame
            df = pd.concat([df, encoded_df], axis=1)
            # Drop the original columns that were one-hot encoded
            df = df.drop(columns=self.columns_categorical_dataencoding)
        elif self.dataencoding_method == "ordinalencoder":
            # Initialize the OrdinalEncoder
            encoder = OrdinalEncoder()
            # Fit and transform the selected columns
            df.loc[:,self.columns_categorical_dataencoding] = encoder.fit_transform(df[self.columns_categorical_dataencoding])
        else:
            print(f"Wrong encoder method provided")

        #################################################################################################################################
        if self.con_scale == True:
            df_scaled = df.copy()
            if self.scale_method == "normalization":
                # Min-max normalization
                scaler_features = MinMaxScaler()
                df_scaled.loc[:,self.columns_features] = scaler_features.fit_transform(df[self.columns_features])
                scaler_target = MinMaxScaler()
                df_scaled.loc[:,self.column_target] = scaler_target.fit_transform(df[self.column_target])
            elif self.scale_method == "standardization":
                # Standardization
                scaler_features = StandardScaler()
                df_scaled.loc[:,self.columns_features] = scaler_features.fit_transform(df[self.columns_features])
                scaler_target = StandardScaler()
                df_scaled.loc[:,self.column_target] = scaler_target.fit_transform(df[self.column_target])
            else:
                print('Worong rescaling method selected, select from normalization/ standardization')

        #################################################################################################################################
        
        print('Data transformation step completed')

        return df_normal, df_scaled, scaler_target, self.scale_method
        
  
class Data_split(BaseEstimator, TransformerMixin):
    """
    A custom data splitter for dividing datasets into training, validation, and test sets. Supports 
    custom split methods including quartile-based, categorical, and k-fold splitting, and handles 
    batching for deep learning models.

    Parameters:
    -----------
    features : list
        List of feature column names.
    target_label : list
        List containing the target column name.
    train_splitsize : float
        Proportion of the dataset to be used for training (between 0 and 1).
    test_splitsize : float
        Proportion of the dataset to be used for testing (between 0 and 1).
    datasplit_seed : int
        Random seed used for reproducibility in data splitting.
    batch_size : int
        Size of batches for training and validation datasets.
    custom_datasplit : bool
        If True, applies a custom data splitting method (quartile, categorical, or k-fold).
    method : dict
        Dictionary defining the data splitting method. The key specifies the method, 
        and the value contains additional parameters.
    columns_pairplots : list
        List of column names to be used for generating pair plots during data splitting.
    columns_contourplots : list
        List of column names to be used for generating contour plots during data splitting.

    Methods:
    --------
    fit(tuple):
        Returns self; no fitting required as this is a stateless transformer.
    
    transform(tuple):
        Applies the following data splitting steps:
        - Splits the dataset into training, validation, and test sets using the specified method.
        - If custom splitting is enabled, uses methods like Quartile, Categorical, or K-fold splitting.
        - Optionally performs data visualization using pair plots and contour plots.
        - Converts data to PyTorch tensors and prepares data loaders for training.

        Returns:
        --------
        - X_train : Torch tensor of training features.
        - Y_train : Torch tensor of training labels.
        - X_val : Torch tensor of validation features.
        - Y_val : Torch tensor of validation labels.
        - X_test : Torch tensor of test features.
        - Y_test : Torch tensor of test labels.
        - scaler_target : The scaling object applied to the target variable.
        - scale_method : The scaling method used.
        - df_normal : The DataFrame before scaling, after splitting.
    """
    
    def __init__(self, features, target_label, train_splitsize, test_splitsize, datasplit_seed, batch_size, custom_datasplit, method, columns_pairplots, columns_contourplots):
        self.features = features
        self.label = target_label
        self.train_splitsize = train_splitsize
        self.test_splitsize = test_splitsize
        self.datasplit_seed = datasplit_seed
        self.batch_size = batch_size
        self.custom_datasplit = custom_datasplit
        self.method = method
        self.columns_pairplots = columns_pairplots
        self.columns_contourplots = columns_contourplots

    def fit(self, tuple):
        return self
       
    def transform(self, tuple):
        #################################################################################################################################

        df_normal = tuple[0]
        df_scaled = tuple[1]
        scaler_target = tuple[2]
        scale_method = tuple[3]

        #################################################################################################################################

        key, value = list(self.method.items())[0]
        if self.custom_datasplit == True:
            if key == 'Quartile':
                train_df, val_df, test_df = custom_data_split_quartile(df_scaled, self.label[0], value, self.train_splitsize, self.test_splitsize, self.datasplit_seed)
            elif key == 'Categorical':
                train_df, val_df, test_df = custom_data_split_categorical(df_scaled, value[0], value[1:], self.train_splitsize, self.test_splitsize, self.datasplit_seed)
            elif key == 'Kfold' or key == 'Custom-Kfold':
                X = torch.tensor(df_scaled[self.features].values.astype(np.float32), dtype=torch.float32)
                Y = torch.tensor(df_scaled[self.label].values.astype(np.float32), dtype=torch.float32)

                # Determine the maximum number of samples that can be evenly distributed
                num_samples = len(X)
                if key == 'Kfold':
                    n_splits = self.method['Kfold']
                elif key == 'Custom-Kfold':
                    n_splits = self.method['Custom-Kfold']
                samples_per_fold = num_samples // n_splits  # Floor division
                usable_samples = samples_per_fold * n_splits

                # Slice data to the usable size
                X = X[:usable_samples]
                Y = Y[:usable_samples]
                X_train_folds, X_val_folds, X_test_folds = [], [], []
                Y_train_folds, Y_val_folds, Y_test_folds = [], [], []
                test_indices = []
                categorical_column = np.array(df_normal['Material'][:usable_samples])
                
                if key == 'Kfold':
                    kf = KFold(n_splits=self.method['Kfold'], shuffle=True, random_state=self.datasplit_seed)
                    folds = list(kf.split(X))
                elif key == 'Custom-Kfold':
                    folds = list(categorical_based_kfold(Y, categorical_column, k=n_splits, random_state=self.datasplit_seed))

                # Iterate through each fold combination
                for i in range(n_splits):
                    # Assign the current fold as the test set
                    test_idx = folds[i][1]
                    X_test, Y_test = X[test_idx], Y[test_idx]
                    
                    # Assign the next fold (cyclically) as the validation set
                    val_idx = folds[(i + 1) % n_splits][1]
                    X_val, Y_val = X[val_idx], Y[val_idx]
                    
                    # Use the remaining folds for training
                    train_idx = np.concatenate(
                        [folds[j][1] for j in range(n_splits) if j != i and j != (i + 1) % n_splits]
                    )
                    X_train, Y_train = X[train_idx], Y[train_idx]
                        
                    Y_train = Y_train.squeeze()
                    Y_val = Y_val.squeeze()
                    Y_test = Y_test.squeeze()

                    # Append splits to their respective lists
                    X_train_folds.append(X_train)
                    X_val_folds.append(X_val)
                    X_test_folds.append(X_test)
                    
                    Y_train_folds.append(Y_train)
                    Y_val_folds.append(Y_val)
                    Y_test_folds.append(Y_test)

                    test_indices.append(test_idx)
                    df_normal = df_normal.reset_index(drop=True)

                # Convert lists to numpy arrays with an additional dimension for the folds
                X_train = torch.tensor(np.array(X_train_folds), dtype=torch.float32)
                X_val = torch.tensor(np.array(X_val_folds), dtype=torch.float32)
                X_test = torch.tensor(np.array(X_test_folds), dtype=torch.float32)

                Y_train = torch.tensor(np.array(Y_train_folds), dtype=torch.float32)
                Y_val = torch.tensor(np.array(Y_val_folds), dtype=torch.float32)
                Y_test = torch.tensor(np.array(Y_test_folds), dtype=torch.float32)
                
                print(f"Number of folds: {n_splits}\nTrain data: {Y_train.shape[1]}\nVal data: {Y_val.shape[1]}\nTest data: {Y_test.shape[1]}")
                
                return X_train, Y_train, X_val, Y_val, X_test, Y_test, test_indices, scaler_target, scale_method, df_normal
        else:
            train_df, temp_df = train_test_split(df_scaled, train_size=self.train_splitsize, shuffle=True, random_state=self.datasplit_seed)
            val_df, test_df = train_test_split(temp_df, test_size=self.test_splitsize, shuffle=True, random_state=self.datasplit_seed)

            print(f"Train data: {len(train_df)}\nVal data: {len(val_df)}\nTest data: {len(test_df)}")
        #################################################################################################################################
        
        pairplots(train_df, val_df, test_df, self.columns_pairplots)
        contourplots(train_df, val_df, test_df, self.columns_contourplots)

        #################################################################################################################################

        X_train = torch.tensor(train_df[self.features].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
        Y_train = torch.tensor(train_df[self.label].values.astype(np.float32), dtype=torch.float32).squeeze().unsqueeze(0)
        X_val = torch.tensor(val_df[self.features].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
        Y_val = torch.tensor(val_df[self.label].values.astype(np.float32), dtype=torch.float32).squeeze().unsqueeze(0)
        X_test = torch.tensor(val_df[self.features].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
        Y_test = torch.tensor(val_df[self.label].values.astype(np.float32), dtype=torch.float32).squeeze()

        #################################################################################################################################

        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, Y_val)
        val_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        print('Data split step completed')
    
        return X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler_target, scale_method, df_normal
        
       
        
        






