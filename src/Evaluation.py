from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
from .CustomFunctions import MinMaxScaler, descale, descale_std

class ErrorMatrix:
    """
    A class to evaluate deterministic and probabilistic models based on various error metrics.

    Attributes:
    -----------
    predictions_lr : np.array
        Predictions from Linear Regression (if deterministic models are considered).
    predictions_randomforest : np.array
        Predictions from Random Forest (if deterministic models are considered).
    predictions_svr : np.array
        Predictions from Support Vector Regression (if deterministic models are considered).
    predictions_knn : np.array
        Predictions from K-Nearest Neighbors (if deterministic models are considered).
    predictions_neuralnet : np.array
        Predictions from Neural Network (if deterministic models are considered).
    predictions_gpr : tuple
        Tuple containing Gaussian Process Regression predictions and variance (if probabilistic models are considered).
    predictions_rfci : tuple
        Tuple containing Random Forest CI predictions and variance (if probabilistic models are considered).
    predictions_pnn : tuple
        Tuple containing Probabilistic Neural Network predictions and variance (if probabilistic models are considered).
    predictions_deepensemble : tuple
        Tuple containing Deep Ensemble predictions and variance (if probabilistic models are considered).
    y_test : np.array
        Ground truth values for evaluation.
    scaler : Scaler object
        Scaler used for inverse transformation if needed.
    con_deter_models : bool
        Flag to indicate if deterministic models are considered.
    con_prob_models : bool
        Flag to indicate if probabilistic models are considered.

    Methods:
    --------
    evaluate(condition_descale: bool) -> dict:
        Evaluates the models using MAE, MSE, and NLL (for probabilistic models).
        
    plot_error(error: dict, label: str, main_directory: str, fold: int, save_plots: bool):
        Plots and optionally saves a bar chart of MAE errors for the models.

    gaussplot_error(error: dict, label: str, main_directory: str, fold: int, save_plots: bool):
        Generates and optionally saves Gaussian distribution plots of model errors.
    """
    def __init__(self, prediction_dic, y_test, scaler_target, scale_method, con_deter_models, con_prob_models):
        if con_deter_models == True:
            self.predictions_lr = prediction_dic['LR'].reshape(-1,1)
            self.predictions_randomforest = prediction_dic['RF'].reshape(-1,1)
            self.predictions_svr = prediction_dic['SVR'].reshape(-1,1)
            self.predictions_knn = prediction_dic['KNN'].reshape(-1,1)
            self.predictions_neuralnet = prediction_dic['NN'].reshape(-1,1)
       
        if con_prob_models == True:
            self.predictions_gpr = prediction_dic['GPR']
            self.predictions_rfci = prediction_dic['RFCI']
            self.predictions_pnn = prediction_dic['PNN']
            self.predictions_deepensemble = prediction_dic['DeepEnsembles']
        
        self.y_test = y_test

        if scale_method == 'standardization':
            self.scaler = scaler_target
        elif scale_method == 'normalization':
            self.scaler = MinMaxScaler(scaler_target)
        
        self.con_deter_models = con_deter_models
        self.con_prob_models = con_prob_models

    def evaluate(self, condition_descale):
        y_test = np.array(self.y_test.unsqueeze(-1))
        
        if self.con_deter_models == True:
            if condition_descale == True:
                y_test = self.scaler.inverse_transform(y_test)
                self.predictions_lr = self.scaler.inverse_transform(self.predictions_lr)
                self.predictions_randomforest = self.scaler.inverse_transform(self.predictions_randomforest)
                self.predictions_svr = self.scaler.inverse_transform(self.predictions_svr)
                self.predictions_knn = self.scaler.inverse_transform(self.predictions_knn)
                self.predictions_neuralnet = self.scaler.inverse_transform(self.predictions_neuralnet)

            mae_lr = mean_absolute_error(y_test, self.predictions_lr)
            mse_lr = mean_squared_error(y_test, self.predictions_lr) 
            
            mae_randomforest = mean_absolute_error(y_test, self.predictions_randomforest)
            mse_randomforest = mean_squared_error(y_test, self.predictions_randomforest)

            mae_svr = mean_absolute_error(y_test, self.predictions_svr)
            mse_svr = mean_squared_error(y_test, self.predictions_svr)

            mae_knn = mean_absolute_error(y_test, self.predictions_knn)
            mse_knn = mean_squared_error(y_test, self.predictions_knn)

            mae_neuralnet = mean_absolute_error(y_test, self.predictions_neuralnet)
            mse_neuralnet = mean_squared_error(y_test, self.predictions_neuralnet)

            error_dic = {'MAE': {'LR': mae_lr, 'RF': mae_randomforest, 'SVR': mae_svr, 'KNN': mae_knn, 'NN': mae_neuralnet} , 'MSE': {'LR': mse_lr, 'RF': mse_randomforest, 'SVR': mse_svr, 'KNN': mse_knn, 'NN': mse_neuralnet}}

        if self.con_prob_models == True:
            if condition_descale == True:
                y_test = self.scaler.inverse_transform(y_test)
                
                predictions_gpr = self.scaler.inverse_transform(self.predictions_gpr[0].reshape(-1,1))
                predictions_rfci = self.scaler.inverse_transform(self.predictions_rfci[0].reshape(-1,1))
                predictions_pnn = self.scaler.inverse_transform(self.predictions_pnn[0].reshape(-1,1))
                predictions_deepensemble = self.scaler.inverse_transform(self.predictions_deepensemble[0].reshape(-1,1))

                var_gpr = np.square(self.predictions_gpr[1] * self.scaler.scale_)
                var_rfci = np.square(self.predictions_rfci[1] * self.scaler.scale_)
                var_pnn = np.square(self.predictions_pnn[1] * self.scaler.scale_)
                var_deepensemble = np.square(self.predictions_deepensemble[1] * self.scaler.scale_)
            else:
                predictions_gpr = self.predictions_gpr[0]
                predictions_rfci = self.predictions_rfci[0]
                predictions_pnn = self.predictions_pnn[0]
                predictions_deepensemble = self.predictions_deepensemble[0]

                var_gpr = np.square(self.predictions_gpr[1])
                var_rfci = np.square(self.predictions_rfci[1])
                var_pnn = np.square(self.predictions_pnn[1])
                var_deepensemble = np.square(self.predictions_deepensemble[1])

            mae_gpr = mean_absolute_error(y_test, predictions_gpr)
            mse_gpr = mean_squared_error(y_test, predictions_gpr)
            nll_gpr = F.gaussian_nll_loss(torch.tensor(predictions_gpr), y_test, torch.tensor(var_gpr))
            mean_std_gpr = np.mean(np.sqrt(var_gpr))

            mae_rfci = mean_absolute_error(y_test, predictions_rfci)
            mse_rfci = mean_squared_error(y_test, predictions_rfci)
            nll_rfci = F.gaussian_nll_loss(torch.tensor(predictions_rfci), y_test, torch.tensor(var_rfci))
            mean_std_rfci = np.mean(np.sqrt(var_rfci))

            mae_pnn = mean_absolute_error(y_test, predictions_pnn)
            mse_pnn = mean_squared_error(y_test, predictions_pnn)
            nll_pnn = F.gaussian_nll_loss(torch.tensor(predictions_pnn), y_test, torch.tensor(var_pnn))
            mean_std_pnn = np.mean(np.sqrt(var_pnn))

            mae_dnn = mean_absolute_error(y_test, predictions_deepensemble)
            mse_dnn = mean_squared_error(y_test, predictions_deepensemble)
            nll_dnn = F.gaussian_nll_loss(torch.tensor(predictions_deepensemble), y_test, torch.tensor(var_deepensemble))
            mean_std_dnn = np.mean(np.sqrt(var_deepensemble))

            error_dic = {'MAE': {'GPR': mae_gpr, 'RFCI': mae_rfci, 'PNN': mae_pnn, 'DeepEnsemble': mae_dnn} , 'MSE': {'GPR': mse_gpr, 'RFCI': mse_rfci, 'PNN': mse_pnn, 'DeepEnsemble': mse_dnn}, 'NLL': {'GPR': nll_gpr.item(), 'RFCI': nll_rfci.item(), 'PNN': nll_pnn.item(), 'DeepEnsemble': nll_dnn.item()}, 'Mean Std': {'GPR': mean_std_gpr, 'RFCI': mean_std_rfci, 'PNN': mean_std_pnn, 'DeepEnsemble': mean_std_dnn}}
        

        if condition_descale == True:
            print('Descaled error')
        else:
            print('Scaled error')

        if self.con_deter_models == True:
            print(f"Mean absolute error: {error_dic['MAE']}\nMean squared error: {error_dic['MSE']}")
        
        if self.con_prob_models == True:
            print(f"Mean absolute error: {error_dic['MAE']}\nNLL: {error_dic['NLL']}\nMean Std: {error_dic['Mean Std']}")

        return error_dic

    def plot_error(self, error, label, main_directory, fold, save_plots):
        # Extract models and their MAE and MSE values
        models = list(error['MAE'].keys())
        mae_values = list(error['MAE'].values())
        mse_values = list(error['MSE'].values())

        if save_plots == True:
            folder_path = os.path.join(main_directory, "02_Plots", "04_Error plots")
            os.makedirs(folder_path, exist_ok=True)

        # Set up the plot
        fig, ax = plt.subplots(figsize=(6, 6))
        bars = ax.bar(models, mae_values, width=0.25, color='skyblue')

        # Add labels, title, and y-axis label
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE')
        ax.set_title(f'Model {label} - Mean absolute error')

        # Add value labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_plots == True:
            file_path = os.path.join(folder_path, f"MAE error_{label}{fold}.png")
            plt.savefig(file_path, format='png', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def gaussplot_error(self, error, label, main_directory, fold, save_plots):
        # Extract models and their MAE and MSE values
        models = list(error['MAE'].keys())
        mae_values = list(error['MAE'].values())
        mse_values = list(error['MSE'].values())

        if save_plots == True:
            folder_path = os.path.join(main_directory, "02_Plots", "04_Error plots")
            os.makedirs(folder_path, exist_ok=True)

        for model, mae in zip(models, mae_values):
            sns.histplot(mae, kde=True)
            plt.title(model)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            mean = np.mean(mae)
            std_dev = np.std(mae)
            plt.text(0.83, 0.93, f'Mean: {mean:.2f}\nStd dev: {std_dev:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
            plt.tight_layout()

            if save_plots == True:
                file_path = os.path.join(folder_path, f"GaussPlot_{model}.png")
                plt.savefig(file_path, format='png', dpi=300)
                plt.close()
            else:
                plt.show()