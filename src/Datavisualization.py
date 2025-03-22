import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import os
from .CustomFunctions import MinMaxScaler
import torch.nn.functional as F
import torch
from sklearn.metrics import mean_absolute_error, r2_score

class Visualization(BaseEstimator, TransformerMixin):
    """
    A custom transformer for visualizing data distributions and relationships.

    Parameters:
    -----------
    columns_visualization : list
        List of columns to visualize (both for scatter and histogram plots).
    columns_pairplots : list
        List of columns to include in pairplot visualizations (not currently used in this method).
    metal_name : str
        The name of the metal or subject for which data is being visualized, used in plot titles.

    Methods:
    --------
    fit(tuple):
        Returns self; no fitting required as this is a stateless transformer.
    
    transform(tuple):
        Visualizes data distributions and relationships using scatter plots and histograms for the specified columns.
        The following steps are applied:
        - Scatter plot of each column against the index to visualize data trends.
        - Histogram and KDE (Kernel Density Estimate) plot of each column to show the data distribution.
        - Displays the mean and standard deviation of each column on the histogram.
        
        The method generates one plot per column and shows them.
        
        Returns:
        --------
        - df_normal : Original, unscaled DataFrame.
        - df_scaled : Scaled DataFrame (if scaling was applied).
        - scaler_target : The scaling object applied to the target variable (if scaling was applied).
        - scale_method : The scaling method used (if scaling was applied).
    """

    def __init__(self, columns_visualization, columns_pairplots, metal_name):
        self.columns_visualization =  columns_visualization
        self.columns_pairplots = columns_pairplots
        self.metal_name = metal_name

    def fit(self, tuple):
        return self
    
    def transform(self, tuple):
        df_normal = tuple[0]
        df_scaled = tuple[1]
        scaler_target = tuple[2]
        scale_method = tuple[3]

        df_plot = df_normal[self.columns_visualization]

        # Plot each column data one by one
        for column in df_plot.columns:
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1)
            sns.scatterplot(x = df_plot.index,y = df_plot[column])
            plt.title(f'{self.metal_name} - {column}', fontsize=8)
            plt.xlabel('Indices')
            plt.ylabel('Value')

            plt.subplot(1,2,2)
            sns.histplot(df_plot[column], kde=True)
            plt.title(f'{self.metal_name} - {column}', fontsize=8)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            mean = df_plot[column].mean()
            std_dev = df_plot[column].std()
            plt.text(0.83, 0.93, f'Mean: {mean:.2f}\nStd dev: {std_dev:.2f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
            plt.tight_layout()
            
            plt.show()

        print('Data visualisation step completed')
        
        return df_normal, df_scaled, scaler_target, scale_method


class InferencePlots():
    """
    A class for generating various plots to visualize and evaluate the performance of predictive models.

    Attributes:
    -----------
    prediction_scaled : dict
        A dictionary of model predictions (scaled).
    y_test : numpy.ndarray
        The true target values for the test set.
    scaler_target : object
        A scaler used for scaling the target values.
    scale_method : str
        The method used for scaling ('standardization' or 'normalization').
    prediction_label : str
        A label describing the prediction variable.
    con_deter_models : bool
        A flag indicating whether to include deterministic models in the plots.
    con_prob_models : bool
        A flag indicating whether to include probabilistic models in the plots.
    con_descale : bool
        A flag indicating whether to descale the predictions and targets.

    Methods:
    --------
    scatterplot():
        Generates scatter plots comparing true values and predicted values from both deterministic and probabilistic models.
    
    residualplot():
        Generates residual plots comparing the true values and errors (residuals) for the deterministic and probabilistic models.
    
    lineplot():
        Generates line plots comparing the true values and predicted values for all models.
    
    errorplot():
        Generates box plots of errors (differences between true and predicted values) across all models.
    
    densityplot():
        Generates density plots (KDE) of true values and predicted values to compare distributions across models.
    
    prob_model_metrix():
        Calculates and displays performance metrics for probabilistic models, including MAE, MSD, NLL, and R² score.
    
    det_model_metrix():
        Calculates and displays performance metrics for deterministic models, including MAE and R² score.
    
    mae_std_plot():
        Generates scatter plots of Mean Absolute Error (MAE) vs Standard Deviation (Std) for probabilistic models.
    """

    def __init__(self, prediction_scaled, y_test, scaler_target, scale_method, prediction_label, con_deter_models, con_prob_models, con_descale):
        """
        Initializes the InferencePlots object with the given parameters.

        Parameters:
        -----------
        prediction_scaled : dict
            A dictionary of model predictions (scaled).
        y_test : numpy.ndarray
            The true target values for the test set.
        scaler_target : object
            A scaler used for scaling the target values.
        scale_method : str
            The method used for scaling ('standardization' or 'normalization').
        prediction_label : str
            A label describing the prediction variable.
        con_deter_models : bool
            A flag indicating whether to include deterministic models in the plots.
        con_prob_models : bool
            A flag indicating whether to include probabilistic models in the plots.
        con_descale : bool
            A flag indicating whether to descale the predictions and targets.
        """

        self.prediction_scaled = prediction_scaled
        self.y_test = y_test.reshape(-1,1)
        self.label = prediction_label
        self.con_deter_models = con_deter_models
        self.con_prob_models = con_prob_models
        self.con_descale = con_descale
        if scale_method == 'standardization':
            self.scaler = scaler_target
        elif scale_method == 'normalization':
            self.scaler = MinMaxScaler(scaler_target)
        self.plot_para = {
                    'axes.titlesize': 26,       # Title font size
                    'axes.labelsize': 26,       # Axis label font size
                    'xtick.labelsize': 26,       # X-axis tick label font size
                    'ytick.labelsize': 26,       # Y-axis tick label font size
                    'legend.fontsize': 22,       # Legend font size
                    'font.size': 26             # General font size
                }
        self.figure_size = (14, 10)

    def scatterplot(self):
        """
        Generates scatter plots comparing the true target values to the predicted values for both deterministic and probabilistic models.

        This method generates separate scatter plots for deterministic models (prediction vs target) and probabilistic models 
        (prediction vs target with 95% confidence intervals). The plots are displayed with tolerance bands at ±10% of the 
        true target values.

        Returns:
        --------
        None
        """

        # Set the style
        sns.set_theme(style="whitegrid")

        if self.con_descale == True:
            y_test = self.scaler.inverse_transform(self.y_test)
        else:
            y_test = self.y_test

        if self.con_deter_models == True:
            for model_name, y_pred in self.prediction_scaled.items():
                if self.con_descale == True:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1)) # 

                plt.rcParams.update( self.plot_para)

                plt.figure(figsize=self.figure_size)
                sns.scatterplot(x=y_test.squeeze(), y=y_pred.squeeze(), s=30, edgecolor="k", alpha=0.8)

                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='k', linestyle='--')
                range_y = max(y_test) - min(y_test)
                percentage = 0.1
                offset = percentage * range_y

                plt.plot([min(y_test), max(y_test)], [min(y_test)+offset, max(y_test)+offset], color='k', linestyle=':', label='Tolerance(10%)')
                plt.plot([min(y_test), max(y_test)], [min(y_test)-offset, max(y_test)-offset], color='k', linestyle=':')
                plt.xlim(min(y_test)-offset*2,max(y_test)+offset*2)
                plt.ylim(min(y_test)-offset*2,max(y_test)+offset*2)
                plt.xlabel(f'Target {self.label}')
                plt.ylabel(f'Predictions {self.label}')
                plt.title(f'Model - {model_name}')
                plt.legend(loc='lower right')
                plt.show()

        if self.con_prob_models == True:
            for model_name, (y_pred, y_std) in self.prediction_scaled.items():
                if self.con_descale == True:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))    # Descale predictions
                    y_std = (y_std * self.scaler.scale_)              # Descale standard deviation
                
                plt.rcParams.update( self.plot_para)

                plt.figure(figsize=self.figure_size)
                sns.scatterplot(x=y_test.squeeze(), y=y_pred.squeeze(), s=30, edgecolor="k", alpha=0.8)              # Scatter plot of predictions

                # Adding confidence intervals
                plt.errorbar(
                    y_test.squeeze(), y_pred.squeeze(), 
                    yerr= 1.96 * y_std,  # 95% confidence interval
                    fmt='o', 
                    alpha=0.3, 
                    label=f'{model_name} CI'
                )

                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='k', linestyle='--')
                range_y = max(y_test) - min(y_test)
                percentage = 0.1
                offset = percentage * range_y

                plt.plot([min(y_test), max(y_test)], [min(y_test)+offset, max(y_test)+offset], color='k', linestyle=':', label='Tolerance(10%)')
                plt.plot([min(y_test), max(y_test)], [min(y_test)-offset, max(y_test)-offset], color='k', linestyle=':')
                plt.xlim(min(y_test)-offset*2,max(y_test)+offset*2)
                plt.ylim(min(y_test)-offset*2,max(y_test)+offset*2)
                plt.xlabel(f'Target {self.label}')
                plt.ylabel(f'Predictions {self.label}')
                plt.title(f'Model - {model_name}')
                plt.legend(loc='lower right')
                plt.show()
    
    def residualplot(self):
        """
        Generates residual plots (errors) comparing the true target values to the predicted values for deterministic and probabilistic models.

        This method displays the residuals (prediction errors) for each model. Residuals are calculated as the difference 
        between the true values and predicted values. Tolerance bands at ±10% of the ideal line are also shown.

        Returns:
        --------
        None
        """
        if self.con_descale == True:
            y_test = self.scaler.inverse_transform(self.y_test)
        else:
            y_test = self.y_test

        if self.con_deter_models == True:
            for model_name, y_pred in self.prediction_scaled.items():
                if self.con_descale == True:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))
                residuals = y_test - y_pred

                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, residuals, alpha=0.7)
                plt.axhline(0, color='k', linestyle='--', label='Ideal Line')
                range_y = max(y_test) - min(y_test)
                percentage = 0.1
                offset = percentage * range_y
                plt.axhline(0+offset, color='k', linestyle=':', label='Tolerance(10%)')
                plt.axhline(0-offset, color='k', linestyle=':')
                plt.xlabel(f'Targets({self.label})')
                plt.ylabel(f'Error')
                plt.title(f'{model_name}')
                plt.legend(loc='lower right')
                plt.show()

        if self.con_prob_models == True:
            for model_name, (y_pred, y_std) in self.prediction_scaled.items():
                if self.con_descale == True:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))    # Descale predictions
                    y_std = (y_std * self.scaler.scale_)              # Descale standard deviation
                residuals = y_test - y_pred

                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, residuals, label=model_name, alpha=0.7)
                plt.axhline(0, color='k', linestyle='--', label='Ideal Line')
                range_y = max(y_test) - min(y_test)
                percentage = 0.1
                offset = percentage * range_y
                plt.axhline(0+offset, color='k', linestyle=':', label='Tolerance(10%)')
                plt.axhline(0-offset, color='k', linestyle=':')
                plt.xlabel(f'Targets({self.label})')
                plt.ylabel(f'Error')
                plt.title(f'{model_name}')
                plt.legend(loc='lower right')
                plt.show()
    
    def lineplot(self):
        """
        Generates line plots comparing the true target values and predicted values for all models.

        This method sorts the true values and plots both the true and predicted values (for deterministic and probabilistic models) 
        as lines. It helps visualize the trends in predictions over the data points.

        Returns:
        --------
        None
        """
        if self.con_descale == True:
            y_test = self.scaler.inverse_transform(self.y_test)
        else:
            y_test = self.y_test
        sorted_indices = np.argsort(y_test)                       # Sort y_test and align predictions accordingly
        y_test_sorted = y_test[sorted_indices]

        plt.figure(figsize=(14, 10))
        plt.plot(y_test_sorted, label='True Values', color='black', linestyle='-', linewidth=2)
        if self.con_deter_models == True:
            for model_name, y_pred in self.prediction_scaled.items():
                if self.con_descale == True:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))
                y_pred = y_pred[sorted_indices]
                plt.plot(y_pred, label=model_name, alpha=0.7, linestyle='--')

        if self.con_prob_models == True:
            for model_name, (y_pred, y_std) in self.prediction_scaled.items():
                if self.con_descale == True:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))    # Descale predictions
                    y_std = (y_std * self.scaler.scale_)              # Descale standard deviation
                y_pred = y_pred[sorted_indices]
                plt.plot(y_pred, label=model_name, alpha=0.7, linestyle='--')

        # Labels and title
        plt.xlabel('Indices')
        plt.ylabel('Prediction')
        plt.title('Model comparison')
        plt.legend(loc='lower right')
        plt.show()

    def errorplot(self):
        """
        Generates box plots to visualize the distribution of errors (differences between the true target values and predicted values) 
        across all models.

        This method calculates the errors for each model and displays them in a box plot format. The box plot provides insight into 
        the spread and central tendency of errors for each model, along with Mean Absolute Error (MAE) for each model.

        Returns:
        --------
        None
        """

        # Set the style
        sns.set_theme(style="white")
        plt.rcParams.update(self.plot_para)

        if self.con_descale == True:
            y_test = self.scaler.inverse_transform(self.y_test)
        else:
            y_test = self.y_test

        # Calculate errors for each model
        errors = {}

        # Loop through the models and their predictions
        if self.con_deter_models == True:
            for model_name, y_pred in self.prediction_scaled.items():
                if self.con_descale == True:
                    # Apply the descale function to y_pred
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))
                
                # Calculate the errors and store them in the dictionary
                errors[model_name] = y_test.squeeze() - y_pred.squeeze()

        if self.con_prob_models == True:
            for model_name, (y_pred, y_std) in self.prediction_scaled.items():
                if self.con_descale == True:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))    # Descale predictions
                errors[model_name] = y_test.squeeze() - y_pred.squeeze()
                
        # Extract errors for each model
        error_data = [errors[model_name] for model_name in self.prediction_scaled.keys()]
        labels = list(self.prediction_scaled.keys())

        # Box plot
        plt.figure(figsize=self.figure_size)
        
        plt.axhline(0, color='k', linestyle='--', label='Ideal Line', linewidth=0.5)
        range_y = max(y_test) - min(y_test)
        percentage = 0.1
        offset = percentage * range_y
        plt.axhline(0+offset, color='k', linestyle=':', label='Tolerance', linewidth=0.5)
        plt.axhline(0-offset, color='k', linestyle=':', linewidth=0.5)

        box = plt.boxplot(error_data, labels=labels, showmeans=True)

        mae = {model_name: abs(errors[model_name]).mean() for model_name in errors.keys()}
        # Prepare MAE string for the box
        mae_text = "Mean Absolute Errors (MAE):\n"
        for model_name, mae_value in mae.items():
            mae_text += f"{model_name}: {mae_value:.2f}\n"

        # Overlay error points
        for i, data in enumerate(error_data, start=1):  # Start index at 1 to match boxplot positions
            x_positions = [i] * len(data)  # Create a list of x-coordinates for scatter plot
            plt.scatter(x_positions, data, alpha=0.6, color='red', label='Error Points' if i == 1 else "")
        
        # Labels and title
        plt.xlabel('Models')
        plt.ylabel(f'Error {self.label}')
        plt.title(f'Error box plot')
        plt.legend(loc='upper left', bbox_to_anchor=(0.78, 1.17), fontsize=18)
        plt.show()
    
    def densityplot(self):
        """
        Generates density plots (Kernel Density Estimate) to visualize the distribution of true target values and predicted values 
        for all models.

        This method plots the true values and predictions from each model in terms of their probability densities, helping to compare 
        the distribution of the predictions and the true values.

        Returns:
        --------
        None
        """
        # Code fo
        if self.con_descale == True:
            y_test = self.scaler.inverse_transform(self.y_test)
        else:
            y_test = self.y_test

        plt.figure(figsize=(14, 10))
        sns.kdeplot(x=y_test.squeeze(), label='True values', fill=False, alpha=0.3, color='black', linewidth=2)
        if self.con_deter_models == True:
            for model_name, y_pred in self.prediction_scaled.items():
                if self.con_descale == True:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))
                sns.kdeplot(x=y_pred.squeeze(), label=model_name, fill=False, alpha=0.3, linewidth=1.5)
        elif self.con_prob_models == True:
            for model_name, (y_pred, y_std) in self.prediction_scaled.items():
                if self.con_descale == True:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))
                sns.kdeplot(x=y_pred.squeeze(), label=model_name, fill=False, alpha=0.3, linewidth=1.5)

        # Labels and title
        plt.xlabel('Targets')
        plt.ylabel('Density')
        plt.title('Prediction Density (All Models)')
        plt.legend(loc='upper right')
        plt.show()

    def prob_model_metrix(self):
        """
        Calculates and displays various performance metrics for probabilistic models, including Mean Absolute Error (MAE), 
        Mean Standard Deviation (MSD), Negative Log-Likelihood (NLL), and R² score.

        This method computes and visualizes the performance metrics for probabilistic models, allowing for the comparison of 
        different models based on multiple evaluation metrics.

        Returns:
        --------
        None
        """
        # Set the style
        sns.set_theme(style="white")
        plt.rcParams.update(self.plot_para)

        if self.con_descale:
            y_test = self.scaler.inverse_transform(self.y_test)
        else:
            y_test = self.y_test

        plt.figure(figsize=self.figure_size)

        # Data for the metrics
        metrics = {"Model": [], "MAE": [], "MSD": [], "NLL": [], 'R2_score': []}

        if self.con_prob_models:
            for model_name, (y_pred, y_std) in self.prediction_scaled.items():
                if self.con_descale:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1))  # Descale predictions
                    y_pred = np.array(y_pred)
                    y_std = (y_std * self.scaler.scale_)  # Descale standard deviation

                y_test = y_test.squeeze()
                y_pred = y_pred.squeeze()

                # Compute MAE
                mae = np.mean(np.abs(y_test - y_pred))

                # Compute R2 score
                r2 = r2_score(y_test, y_pred)

                # Compute Mean Standard Deviation (MSD)
                msd = np.mean(y_std)

                y_var = np.square(y_std)
                nll = F.gaussian_nll_loss(torch.tensor(y_pred), torch.tensor(y_test), torch.tensor(y_var))

                # Append to metrics
                metrics["Model"].append(model_name)
                metrics["MAE"].append(mae)
                metrics["MSD"].append(msd)
                metrics["NLL"].append(nll.item())  # Extract value from tensor
                metrics['R2_score'].append(r2)

            # Convert metrics to numpy arrays
            x = np.arange(len(metrics["Model"]))  # the label locations
            width = 0.2  # the width of the bars

            fig, ax1 = plt.subplots(figsize=self.figure_size)

            # Bars for MAE, MSD, and NLL
            mae_bars = ax1.bar(x - width, metrics["MAE"], width, label="Mean Absolute Error (MAE)", color="tab:blue")
            msd_bars = ax1.bar(x, metrics["MSD"], width, label="Mean Standard Deviation (MSD)", color="tab:orange")
            nll_bars = ax1.bar(x + width, metrics["NLL"], width, label="Negative Log-Likelihood (NLL)", color="tab:green")

            # Create another y-axis for R² score
            ax2 = ax1.twinx()
            r2_bars = ax2.bar(
                x + 2 * width, metrics["R2_score"], width, label="R² Score", color="tab:red", alpha=0.7
            )

            # Annotate the bars with their values
            def annotate_bars(bars, ax):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{height:.2f}', ha='center', va='bottom', fontsize=16)

            annotate_bars(mae_bars, ax1)
            annotate_bars(msd_bars, ax1)
            annotate_bars(nll_bars, ax1)
            annotate_bars(r2_bars, ax2)

            # Set labels and titles
            ax1.set_xlabel("Models")
            ax1.set_ylabel("Metric Values (MAE, MSD, NLL)", color="black")
            ax1.tick_params(axis='y', labelcolor="black")
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics["Model"])

            ax2.set_ylabel("R² Score", color="tab:red")
            ax2.tick_params(axis='y', labelcolor="tab:red")

            # Combine legends from both axes
            bars_labels = [mae_bars, msd_bars, nll_bars, r2_bars]
            bars_legend = [bar.get_label() for bar in bars_labels]

            ax1.legend(
                bars_labels,
                bars_legend,
                loc='upper left',
                bbox_to_anchor=(0.75, 1.15),  # Position legend outside plot area
                borderaxespad=0,
                fontsize=12
            )

            plt.title("Model Performance Metrics")
            fig.tight_layout()
            plt.show()

    def det_model_metrix(self):
        """
        Calculates and displays performance metrics for deterministic models, including Mean Absolute Error (MAE) and R² score.

        This method calculates the MAE and R² score for each deterministic model and visualizes the comparison of these metrics 
        across all models using bar charts.

        Returns:
        --------
        None
        """
        # Set the style
        sns.set_theme(style="white")
        plt.rcParams.update(self.plot_para)

        if self.con_descale == True:
                y_test = self.scaler.inverse_transform(self.y_test)
        else:
            y_test = self.y_test
        y_test =np.array(y_test)

        # Initialize metrics dictionary
        metrics = {"Model": [], "MAE": [], "R2": []}

        # Loop through each model and compute metrics
        for model_name, y_pred in self.prediction_scaled.items():
            if self.con_descale == True:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))

            # Compute metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store metrics
            metrics["Model"].append(model_name)
            metrics["MAE"].append(mae)
            metrics["R2"].append(r2)

       # Plot grouped bar graph for MAE with R^2 on a secondary y-axis
        bar_width = 0.25
        bar_positions = np.arange(len(metrics["Model"]))

        fig, ax1 = plt.subplots(figsize=self.figure_size)

        # MAE bars
        mae_bars = ax1.bar(bar_positions - bar_width / 2, metrics["MAE"], bar_width, color="tab:blue", label="MAE", alpha=1)  # 
        ax1.set_ylabel("MAE", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor="tab:blue")

        # Secondary y-axis for R^2
        ax2 = ax1.twinx()
        r2_bars = ax2.bar(bar_positions + bar_width / 2, metrics["R2"], bar_width, color="tab:orange", label="R2", alpha=1) # 
        ax2.set_ylabel("R2 Score", color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")

        # Add values on top of MAE bars
        for bar in mae_bars:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{bar.get_height():.2f}",
                ha="center",
                fontsize=16,
                color="black",
            )

        # Add values on top of R^2 bars
        for bar in r2_bars:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2f}",
                ha="center",
                fontsize=16,
                color="black",
            )
        
        # Add labels, title, and legend
        ax1.set_xticks(bar_positions)
        ax1.set_xticklabels(metrics["Model"], ha="right")
        ax1.set_xlabel('Models')
        plt.title("Model Performance Metrics")

        # Adding legends to a single box for the whole figure
        lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]  # Unpack and concatenate handles and labels

        # Place the legend in the available space
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.8, 0.98), ncol=2, fontsize=16)

        plt.tight_layout()
        plt.show()

    def mae_std_plot(self):
        """
        Generates scatter plots of Mean Absolute Error (MAE) vs Standard Deviation (Std) for probabilistic models.

        This method plots the MAE against the predicted standard deviation (Std) for each probabilistic model, helping to identify 
        how prediction uncertainty (represented by the standard deviation) correlates with the error magnitude.

        Returns:
        --------
        None
        """
        if self.con_descale == True:
            y_test = self.scaler.inverse_transform(self.y_test)
        else:
            y_test = self.y_test

        if self.con_prob_models == True:
            for model_name, (y_pred, y_std) in self.prediction_scaled.items():
                if self.con_descale == True:
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1,1))    # Descale predictions
                    y_std = (y_std * self.scaler.scale_)              # Descale standard deviation

                mae = abs(y_test - y_pred)
                plt.figure(figsize=(14, 10))
                plt.scatter(mae.squeeze(), y_std.squeeze(), alpha=0.6)              # Scatter plot of predictions
                plt.xlabel('MAE')
                plt.ylabel('Std')
                plt.title(f"{model_name}")
                plt.show()

         
def plot_scatter_with_categories(df, x_col, y_col, category_col, title):
    """
    Plots a scatter plot with different colors for categories in a specified column.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - x_col (str): Name of the column to be used for the x-axis.
    - y_col (str): Name of the column to be used for the y-axis.
    - category_col (str): Name of the column containing categorical data.
    - title (str): Title of the scatter plot.
    """
    # Set the style
    sns.set_theme(style="whitegrid")
    plot_para = {
                    'axes.titlesize': 26,       # Title font size
                    'axes.labelsize': 26,       # Axis label font size
                    'xtick.labelsize': 26,       # X-axis tick label font size
                    'ytick.labelsize': 26,       # Y-axis tick label font size
                    'legend.fontsize': 22,       # Legend font size
                    'font.size': 26             # General font size
                }
    plt.rcParams.update(plot_para)
    
    # Create the scatter plot
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=category_col,
        palette="tab10",  # Use a distinct color palette for categories
        s=30,           # Marker size
        edgecolor="k",   # Marker edge color
        alpha=0.8        # Transparency
    )
    
    y_test = df[x_col]
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='k', linestyle='--')
    range_y = max(y_test) - min(y_test)
    percentage = 0.1
    offset = percentage * range_y
    plt.plot([min(y_test), max(y_test)], [min(y_test)+offset, max(y_test)+offset], color='k', linestyle=':', label='Tolerance(10%)')
    plt.plot([min(y_test), max(y_test)], [min(y_test)-offset, max(y_test)-offset], color='k', linestyle=':')
    plt.xlim(min(y_test)-offset*2,max(y_test)+offset*2)
    plt.ylim(min(y_test)-offset*2,max(y_test)+offset*2)
    
    plt.rcParams.update(plot_para)
    
    # Customize the plot
    plt.title(f'Model - {title}')
    plt.xlabel('Target $S_d$ (MPa)')
    plt.ylabel('Predictions $S_d$ (MPa)')
    plt.legend()
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def plot_scatter_with_categories_and_std(df, x_col, y_col, std_col, category_col, title):
    """
    Plots a scatter plot with different colors for categories in a specified column,
    including standard deviation lines for error bars.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - x_col (str): Name of the column to be used for the x-axis.
    - y_col (str): Name of the column to be used for the y-axis.
    - category_col (str): Name of the column containing categorical data.
    - std_col (str): Name of the column containing standard deviation values.
    - title (str): Title of the scatter plot.
    """
    # Set the style
    sns.set_theme(style="whitegrid")
    plot_para = {
                    'axes.titlesize': 26,       # Title font size
                    'axes.labelsize': 26,       # Axis label font size
                    'xtick.labelsize': 26,       # X-axis tick label font size
                    'ytick.labelsize': 26,       # Y-axis tick label font size
                    'legend.fontsize': 22,       # Legend font size
                    'font.size': 26             # General font size
                }
    plt.rcParams.update(plot_para)

    plt.figure(figsize=(14, 10))

    # Iterate through each category to plot individually
    categories = df[category_col].unique()
    for category in categories:
        category_data = df[df[category_col] == category]
        x_data = category_data[x_col]
        y_data = category_data[y_col]
        y_std = category_data[std_col]

        # Scatter plot for each category
        plt.scatter(
            x_data,
            y_data,
            label=f"{category}",
            s=30,
            alpha=0.8,
            edgecolor="k"
        )

        # Plot error bars for standard deviation
        plt.errorbar(
            x_data,
            y_data,
            yerr=1.96 * y_std,  # 95% confidence interval
            fmt='o',
            alpha=0.3,
            label=f"{category} (CI)"
        )

    # Add diagonal and tolerance lines
    y_test = df[x_col]
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='k', linestyle='--')
    range_y = max(y_test) - min(y_test)
    percentage = 0.1
    offset = percentage * range_y
    plt.plot([min(y_test), max(y_test)], [min(y_test)+offset, max(y_test)+offset], color='k', linestyle=':', label='Tolerance(10%)')
    plt.plot([min(y_test), max(y_test)], [min(y_test)-offset, max(y_test)-offset], color='k', linestyle=':')
    plt.xlim(min(y_test)-offset*2,max(y_test)+offset*2)
    plt.ylim(min(y_test)-offset*2,max(y_test)+offset*2)
    
    # Customize the plot
    plt.title(f'Model - {title}')
    plt.xlabel('Target $S_d$ (MPa)')
    plt.ylabel('Predictions $S_d$ (MPa)')
    plt.legend()
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def plot_hypopt_loss_scatter(total_losses, title, label):
    """
    Plots a scatter plot of total loss.

    Parameters:
    - total_losses (list or array): List of total loss values.
    - x_values (list or array): X-axis values corresponding to the total losses.
                                If None, indices of total_losses will be used.
    - title (str): Title of the scatter plot.
    """
    # Set the style
    sns.set_theme(style="whitegrid")

    x_values = range(len(total_losses))  # Use indices if no x_values are provided

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=x_values, y=total_losses, s=30, edgecolor="k", alpha=0.8)

    # Add labels, title, and grid
    plt.title(title, fontsize=16)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel(label, fontsize=14)
    # plt.grid(True, linestyle="--", alpha=0.5)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_heatmap_mae(df, input_col1, input_col2, target, predictions, ylabel, xlabel, cbar_label, bin_size, cmap="viridis"):
    """
    Plots a heatmap for the relationship between two input columns and an output column,
    with a labeled color bar and controlled bin size. The axis labels will show the bin ranges.

    Args:
        df (pd.DataFrame): The input DataFrame.
        input_col1 (str): Name of the first input column.
        input_col2 (str): Name of the second input column.
        target (str): Name of the target column.
        predictions (str): Name of the predictions column.
        cbar_label (str): Label for the color bar.
        cmap (str): Colormap for the heatmap (default is 'viridis').
        bin_size (int): Number of bins for discretizing input columns (default is 10).
    """
    # Set the style
    sns.set_theme(style="white")

    # Bin the features into fewer intervals (e.g., 10 bins) and get bin edges
    feature1_bins = pd.cut(df[input_col1], bins=bin_size, retbins=True)
    feature2_bins = pd.cut(df[input_col2], bins=bin_size, retbins=True)

    # Assign binned labels to features
    df['Feature1_binned'] = feature1_bins[0]
    df['Feature2_binned'] = feature2_bins[0]

    df['error'] = abs(df[target] - df[predictions])
    
    # Pivot table to aggregate target values (e.g., mean target value for binned feature combinations)
    heatmap_data_mae = df.pivot_table(
        index='Feature1_binned', 
        columns='Feature2_binned', 
        values='error', 
        aggfunc='mean'
    )

    # Plot heatmap
    plt.figure(figsize=(14, 10))
    plt.rcParams.update({
                        'axes.titlesize': 24,
                        'axes.labelsize': 24,
                        'xtick.labelsize': 20,
                        'ytick.labelsize': 20,
                        'legend.fontsize': 24,
                        'font.size': 20
                    })
    
    # Plot MAE heatmap
    ax1 = sns.heatmap(
        heatmap_data_mae, 
        annot=True, 
        cmap=cmap, 
        fmt=".2f", 
        cbar=True, 
        square=False, 
        linewidths=0.5, 
        linecolor='black',
    )
    #ax1.set_title("Mean Absolute Error Heatmap")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    cbar1 = ax1.collections[0].colorbar
    cbar1.set_label(cbar_label)
    
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    # Add black border around the heatmap
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    #plt.tight_layout()
    plt.show()

def plot_heatmap_count(df, input_col1, input_col2, target, predictions, ylabel, xlabel, cbar_label, bin_size, cmap="viridis"):
    """
    Plots a heatmap for the relationship between two input columns and an output column,
    with a labeled color bar and controlled bin size. The axis labels will show the bin ranges.

    Args:
        df (pd.DataFrame): The input DataFrame.
        input_col1 (str): Name of the first input column.
        input_col2 (str): Name of the second input column.
        target (str): Name of the target column.
        predictions (str): Name of the predictions column.
        cbar_label (str): Label for the color bar.
        cmap (str): Colormap for the heatmap (default is 'viridis').
        bin_size (int): Number of bins for discretizing input columns (default is 10).
    """
    # Set the style
    sns.set_theme(style="white")

    # Bin the features into fewer intervals (e.g., 10 bins) and get bin edges
    feature1_bins = pd.cut(df[input_col1], bins=bin_size, retbins=True)
    feature2_bins = pd.cut(df[input_col2], bins=bin_size, retbins=True)

    # Assign binned labels to features
    df['Feature1_binned'] = feature1_bins[0]
    df['Feature2_binned'] = feature2_bins[0]

    df['error'] = abs(df[target] - df[predictions])
    
    # Pivot table to aggregate target values (e.g., mean target value for binned feature combinations)
    heatmap_data_count = df.pivot_table(
        index='Feature1_binned', 
        columns='Feature2_binned', 
        values='error', 
        aggfunc='count'
    )

    # Remove rows and columns where all values are zero
    heatmap_data_count = heatmap_data_count.loc[(heatmap_data_count != 0).any(axis=1), (heatmap_data_count != 0).any(axis=0)]

    # Plot heatmap
    plt.figure(figsize=(14, 10))
    plt.rcParams.update({
                        'axes.titlesize': 24,
                        'axes.labelsize': 24,
                        'xtick.labelsize': 20,
                        'ytick.labelsize': 20,
                        'legend.fontsize': 24,
                        'font.size': 20
                    })
    
    # Plot MAE heatmap
    ax1 = sns.heatmap(
        heatmap_data_count, 
        annot=True, 
        cmap=cmap, 
        fmt=".0f", 
        cbar=True, 
        square=False, 
        linewidths=0.5, 
        linecolor='black',
    )
    #ax1.set_title("Mean Absolute Error Heatmap")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    cbar1 = ax1.collections[0].colorbar
    cbar1.set_label(cbar_label)
    
    # Set x-axis ticks to 0 degrees
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    # Add black border around the heatmap
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    #plt.tight_layout()
    plt.show()
    
def plot_heatmap(df, input_col1, input_col2, target, ylabel, xlabel, cbar_label, bin_size, cmap="viridis"):
    # Bin the features into fewer intervals (e.g., 10 bins) and get bin edges
    feature1_bins = pd.cut(df[input_col1], bins=bin_size, retbins=True)
    feature2_bins = pd.cut(df[input_col2], bins=bin_size, retbins=True)

    # Assign binned labels to features
    df['Feature1_binned'] = feature1_bins[0]
    df['Feature2_binned'] = feature2_bins[0]

    # Get bin edges for labeling
    # feature1_labels = [f"{int(left)}-{int(right)}" for left, right in zip(feature1_bins[1][:-1], feature1_bins[1][1:])]
    # feature2_labels = [f"{int(left)}-{int(right)}" for left, right in zip(feature2_bins[1][:-1], feature2_bins[1][1:])]

    # Pivot table to aggregate target values (e.g., mean target value for binned feature combinations)
    heatmap_data = df.pivot_table(
        index='Feature1_binned', 
        columns='Feature2_binned', 
        values=target, 
        aggfunc='mean'
    )

    # Plot heatmap
    plt.figure(figsize=(14, 12))
    plt.rcParams.update({
                        'axes.titlesize': 24,
                        'axes.labelsize': 24,
                        'xtick.labelsize': 20,
                        'ytick.labelsize': 20,
                        'legend.fontsize': 24,
                        'font.size': 20
                    })
    ax = sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap=cmap, 
        fmt=".2f", 
        cbar=True, 
        square=True, 
        linewidths=0.5, 
        linecolor='black'
    )


    # Add black border around the heatmap
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label)

    # Titles and labels
    #plt.title("Heatmap with Real Bin Ranges on Axes")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(rotation=0)
    plt.show()
