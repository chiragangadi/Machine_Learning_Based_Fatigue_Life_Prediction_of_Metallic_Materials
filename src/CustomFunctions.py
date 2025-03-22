import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import KFold
import torch
from sklearn.gaussian_process.kernels import Kernel
from sklearn.utils import shuffle

class MinMaxScaler():
    def __init__(self, scaler):
        self.min = scaler.data_min_
        self.max = scaler.data_max_
        self.scale_ = self.max - self.min

    def inverse_transform(self, predictions):
        return (predictions*self.scale_) + self.min

def descale(y, y_min, y_max):
    y = np.array(y)
    return  y * (y_max - y_min) + y_min

def descale_std(y, y_min, y_max):
    y = np.array(y)
    return  y * (y_max - y_min)

def custom_data_split_quartile(df, column, quartile_list, train_size, test_size, datasplit_seed):
    """
    Split a DataFrame into train, validation, and test sets based on quartiles.

    Parameters:
        df (pd.DataFrame): The DataFrame to split.
        column (str): The column used for quartile-based splitting.
        train_size (float): Proportion of data to include in the train set.
        val_size (float): Proportion of data to include in the validation set.
        test_size (float): Proportion of data to include in the test set.
        random_state (int): Random state for reproducibility.

    Returns:
        pd.DataFrame: Train, validation, and test DataFrames.
    """

    # Initialize empty lists to hold each set
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Calculate quartiles
    quartiles = np.quantile(df[column], quartile_list)
    
    # Loop through each quartile and split
    for i in range(len(quartile_list)-1):
        # Filter data for the current quartile
        quartile_data = df[(df[column] >= quartiles[i]) & (df[column] < quartiles[i+1])]

        # Split data into train, validation, and test for the current quartile
        train_quartile, temp_quartile = train_test_split(quartile_data, train_size=train_size, shuffle=True, random_state=datasplit_seed)
        val_quartile, test_quartile = train_test_split(temp_quartile, test_size=test_size, shuffle=True, random_state=datasplit_seed)
        

        # Append splits to corresponding lists
        train_df = pd.concat((train_df, train_quartile), axis=0).reset_index(drop=True)
        val_df = pd.concat((val_df, val_quartile), axis=0).reset_index(drop=True)
        test_df = pd.concat((test_df, test_quartile), axis=0).reset_index(drop=True)

    return train_df, val_df, test_df

def custom_data_split_categorical(df, column, string_list, train_size, test_size, datasplit_seed):
    # Initialize empty lists to hold each set
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for i in range(len(string_list)):
        df_sub = pd.DataFrame()
        df_sub = df[df[column]==string_list[i]]
        df_train, df_temp = train_test_split(df_sub, train_size=train_size, shuffle=True, random_state=datasplit_seed)
        df_val, df_test = train_test_split(df_temp, test_size=train_size, shuffle=True, random_state=datasplit_seed)

        # Append splits to corresponding lists
        train_df = pd.concat((train_df, df_train), axis=0).reset_index(drop=True)
        val_df = pd.concat((val_df, df_val), axis=0).reset_index(drop=True)
        test_df = pd.concat((test_df, df_test), axis=0).reset_index(drop=True)
    
    return train_df, val_df, test_df

def quartile_based_kfold(data, k):
    """
    Custom K-Fold splitting ensuring each fold has a balanced representation of quartiles
    and no repetition of data in test sets.
    
    Parameters:
    - data: List or numpy array of the data to split.
    - k: Number of folds.
    
    Returns:
    - A generator yielding train and test indices for each fold.
    """
    # Sort the data to calculate quartiles
    sorted_data = np.sort(data)
    n = len(sorted_data)

    # Calculate the quartiles
    Q1 = np.percentile(sorted_data, 25)
    Q3 = np.percentile(sorted_data, 75)

    # Split the data into quartiles
    lower_quartile = sorted_data[sorted_data <= Q1]
    middle_quartile = sorted_data[(sorted_data > Q1) & (sorted_data <= Q3)]
    upper_quartile = sorted_data[sorted_data > Q3]

    # Create K-Folds for each quartile
    kf = KFold(n_splits=k, shuffle=False)

    # Indices for each quartile
    lower_indices = np.where(np.isin(data, lower_quartile))[0]
    middle_indices = np.where(np.isin(data, middle_quartile))[0]
    upper_indices = np.where(np.isin(data, upper_quartile))[0]

    np.random.seed(42)
    # Split each quartile into k subgroups
    lower_splits = np.array_split(k, lower_indices)
    middle_splits = np.array_split(k, middle_indices)
    upper_splits = np.array_split(k, upper_indices)

    # Generate k folds without repetition of test data
    for i in range(k):
        # Combine one group from each quartile to form the test set for this fold
        test_indices = np.concatenate([lower_splits[i], middle_splits[i], upper_splits[i]])

        # Combine all other groups (from the remaining quartiles) to form the training set
        # Exclude the current fold's indices from the other quartiles
        train_indices = np.concatenate([np.concatenate([lower_splits[j] for j in range(k) if j != i]),
                                        np.concatenate([middle_splits[j] for j in range(k) if j != i]),
                                        np.concatenate([upper_splits[j] for j in range(k) if j != i])])

        # Flatten the train_indices array
        train_indices = train_indices.flatten()

        # Yield train and test indices for this fold
        yield train_indices, test_indices

def categorical_based_kfold(y, categorical_column, k, random_state):
        """
        Custom K-fold function that ensures homogeneity by creating subsets based
        on classes in a categorical column and concatenating them to
        form final folds.

        Parameters:
            y (numpy array): The target array with values.
            categorical_column (numpy array): The categorical column to base the folds on.
            k (int): The number of folds.
            random_state (int, optional): Seed for reproducibility.

        Returns:
            list: A list of tuples, where each tuple contains train and test indices as numpy arrays.
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Find unique categories and their indices
        unique_categories, category_indices = np.unique(categorical_column, return_inverse=True)
        category_folds = {cat: [] for cat in unique_categories}
        
        # Create indices for each category
        for cat in unique_categories:
            cat_indices = np.where(category_indices == cat)[0]
            cat_indices = shuffle(cat_indices, random_state=random_state)
            split_size = len(cat_indices) // k

            # Split indices for this category into folds
            for fold in range(k):
                start = fold * split_size
                end = (fold + 1) * split_size if fold < k - 1 else len(cat_indices)
                category_folds[cat].append(cat_indices[start:end])

        # Combine category-specific folds into overall folds
        folds = []
        for fold in range(k):
            test_indices = np.concatenate([category_folds[cat][fold] for cat in unique_categories])
            train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
            folds.append((train_indices, test_indices))
        
        return folds

def handle_outliers_zscore(df, columns, method, factor):
    """
    Handle outliers in specific columns using the Z-score method, with options to remove or cap outliers.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of columns to apply outlier handling.
        factor (float): Z-score threshold for detecting outliers. Default is 3.
        method (str): 'remove' to drop outliers, 'cap' to cap outliers. Default is 'remove'.

    Returns:
        pd.DataFrame: DataFrame with outliers handled in specified columns.
    """
    df_copy = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    z_scores = np.abs((df_copy[columns] - df_copy[columns].mean()) / df_copy[columns].std())

    if method == "remove":
        # Remove rows where any specified column has a Z-score above the threshold
        df_copy = df_copy[(z_scores < factor).all(axis=1)]
    
    elif method == "cap":
        # Cap values to the threshold for outliers
        for col in columns:
            upper_limit = df_copy[col].mean() + factor * df_copy[col].std()
            lower_limit = df_copy[col].mean() - factor * df_copy[col].std()
            df_copy[col] = np.where(df_copy[col] > upper_limit, upper_limit,
                           np.where(df_copy[col] < lower_limit, lower_limit, df_copy[col]))
    
    return df_copy

def handle_outliers_iqr(df, columns, method, factor):
    """
    Handle outliers in specific columns using the IQR method, with options to remove or cap outliers.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        columns (list): List of columns to apply outlier handling.
        method (str): 'remove' to drop outliers, 'cap' to cap outliers. Default is 'remove'.

    Returns:
        pd.DataFrame: DataFrame with outliers handled in specified columns.
    """
    df_copy = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    Q1 = df_copy[columns].quantile(0.25)
    Q3 = df_copy[columns].quantile(0.75)
    IQR = Q3 - Q1

    if method == "remove":
        # Remove rows where any specified column has an outlier based on IQR
        mask = ~((df_copy[columns] < (Q1 - factor * IQR)) | (df_copy[columns] > (Q3 + factor * IQR))).any(axis=1)
        df_copy = df_copy[mask]
    
    elif method == "cap":
        # Cap values to the IQR limits for outliers
        for col in columns:
            lower_limit = Q1[col] - factor * IQR[col]
            upper_limit = Q3[col] + factor * IQR[col]
            df_copy[col] = np.where(df_copy[col] > upper_limit, upper_limit,
                        np.where(df_copy[col] < lower_limit, lower_limit, df_copy[col]))

    return df_copy

def pairplots(df1, df2, df3, columns_pairplots):

    # Add a label column to each DataFrame
    df1['Dataset'] = 'Training data'
    df2['Dataset'] = 'Validation data'
    df3['Dataset'] = 'Test data'

    # Concatenate the DataFrames
    df_combined = pd.concat([df1, df2, df3])

    for name, col in columns_pairplots.items():
        sns.pairplot(df_combined, vars=col, hue='Dataset')
        plt.show()

def contourplots(df1, df2, df3, columns_pairplots):
    
    # Add a label column to each DataFrame
    df1['Dataset'] = 'Training data'
    df2['Dataset'] = 'Validation data'
    df3['Dataset'] = 'Test data'

    # Concatenate the DataFrames
    df_combined = pd.concat([df1, df2, df3])

    for name, col in columns_pairplots.items():
        # Set up the plot
        plt.figure(figsize=(8, 6))

        # Define color maps for each dataset
        colors = {'Training data': 'Blues', 'Validation data': 'Oranges', 'Test data': "Greens"}

        # Loop through each dataset and create a contour plot for each
        for label, df_subset in df_combined.groupby('Dataset'):
            sns.kdeplot(
                x=df_subset[col[0]], y=df_subset[col[1]], 
                fill=False, cmap=colors[label], label=label,
                alpha=0.5, thresh=0.05
            )

        # Manually create legend handles
        handles = [
            mpatches.Patch(color='skyblue', label='Training data'),
            mpatches.Patch(color='orange', label='Validation data'),
            mpatches.Patch(color='green', label='Test data')
        ]

        # Add the legend with manually created handles
        plt.legend(handles=handles, title='Dataset')

        # Add legend and labels
        plt.xlabel(f'{col[0]}')
        plt.ylabel(f'{col[1]}')
        plt.title('Contour Plot')
        plt.show()

class FeatureDotProduct(Kernel):
    def __init__(self, sigma_0):
        self.sigma_0 = np.asarray(sigma_0)

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        # Compute the kernel matrix
        K = X @ Y.T + np.sum(self.sigma_0)

        if eval_gradient:
            # Gradient with respect to sigma_0
            # Since sigma_0 only affects the additive term, the gradient is a matrix of ones
            gradient = np.ones((X.shape[0], Y.shape[0], len(self.sigma_0)))
            return K, gradient

        return K

    def diag(self, X):
        return np.einsum('ij,ij->i', X, X) + np.sum(self.sigma_0)

    def is_stationary(self):
        return False





