from sklearn.base import BaseEstimator, TransformerMixin
from .MachineLearning_models import ML_Models, TransferLearningModel
from .Datavisualization import InferencePlots
from .Evaluation import ErrorMatrix
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from .CustomFunctions import MinMaxScaler, quartile_based_kfold

class ML_Models_Train(BaseEstimator, TransformerMixin):
    """
    A custom transformer class that trains and evaluates machine learning models. This class supports 
    deterministic (e.g., Random Forest, SVR) and probabilistic models (e.g., GPR, RFCI) with 
    cross-validation, hyperparameter optimization, and error evaluation. It also handles scaling 
    and the option to perform different training and prediction strategies.

    Attributes:
    -----------
    method : dict
        A dictionary specifying the method for cross-validation ('Kfold' or 'Custom-Kfold').
    target_label : str
        The name of the target variable/label in the dataset.
    custom_split : bool
        Whether to use a custom data split strategy.
    seed : int
        The random seed for reproducibility.
    con_hypopt : bool
        Whether to perform hyperparameter optimization for the models.
    con_deter_models : bool
        Whether to train deterministic models (e.g., Random Forest, SVR).
    con_probmodels : bool
        Whether to train probabilistic models (e.g., GPR, RFCI).
    con_descale : bool
        Whether to apply descaling to the predicted values.

    Methods:
    --------
    fit(inputs):
        A placeholder function to comply with scikit-learn's API.
        Does not train any models.

    transform(inputs):
        Transforms the input data by training and evaluating models with cross-validation.
        Supports both deterministic and probabilistic models, scaling, and error evaluation.
        
        Parameters:
        -----------
        inputs : list
            A list containing the following elements:
            1. x_train : ndarray
                Training feature data.
            2. y_train : ndarray
                Training target labels.
            3. x_val : ndarray
                Validation feature data.
            4. y_val : ndarray
                Validation target labels.
            5. x_test : ndarray
                Testing feature data.
            6. y_test_ : ndarray
                Testing target labels.
            7. test_indices : list
                Indices of the test data used during k-fold testing.
            8. scaler_target : object
                The scaler used for target variable scaling.
            9. scale_method : str
                The scaling method used, either 'standardization' or 'normalization'.
            10. df_normal : DataFrame
                DataFrame to store the predictions and evaluation metrics.

        Returns:
        --------
        models : ML_Models
            The trained model(s) after the fitting process.
        best_params_det : dict
            A dictionary containing the best hyperparameters for deterministic models.
        best_params_prob : dict
            A dictionary containing the best hyperparameters for probabilistic models.
        df_normal : DataFrame
            A DataFrame containing the predictions and evaluation metrics from the models.
    """
    def __init__(self, method, target_label, custom_split, seed, con_hypopt, con_deter_models, con_prob_models, con_descale):
        self.method = method
        self.target_label = target_label
        self.custom_split = custom_split
        self.seed = seed
        self.con_hypopt = con_hypopt
        self.con_deter_models = con_deter_models
        self.con_probmodels = con_prob_models
        self.con_descale = con_descale

    def fit(self, inputs):
        return self
        
    def transform(self, inputs):
        key, value = list(self.method.items())[0]

        if (key == 'Kfold' or key == 'Custom-Kfold') and self.custom_split == True:
            x_train, y_train, x_val, y_val, x_test, y_test_, test_indices, scaler_target, scale_method, df_normal = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9]
            best_params_det = {'RF': None, 'SVR':None, 'KNN':None, 'NN':None}
            best_params_prob = {'GPR':None, 'RFCI':None, 'PNN':None, 'DeepEnsembles':None}

            if scale_method == 'standardization':
                scaler = scaler_target
            elif scale_method == 'normalization':
                scaler = MinMaxScaler(scaler_target)

            randomforest_mae, svr_mae, knn_mae, neuralnet_mae = [], [], [], []
            allpred_lr, allpred_randomforest, allpred_svr, allpred_knn, allpred_neuralnet, y_test = [], [], [], [], [], []
            mean_gpr, mean_rfci, mean_pnn, mean_denn, std_gpr, std_rfci, std_pnn, std_denn = [], [], [], [], [], [], [], []

            if self.con_deter_models == True:
                # Initialize the new columns in the dataframe
                df_normal['Pred_LR'] = np.nan  # Linear Regression Predictions
                df_normal['Pred_RF'] = np.nan  # Random Forest Predictions
                df_normal['Pred_SVR'] = np.nan  # SVR Predictions
                df_normal['Pred_KNN'] = np.nan  # KNN Predictions
                df_normal['Pred_NN'] = np.nan  # Neural Net Predictions

            if self.con_probmodels == True:
                df_normal['Pred_mean_GPR'] = np.nan  # Linear Regression Predictions
                df_normal['Pred_mean_RFCI'] = np.nan  # Random Forest Predictions
                df_normal['Pred_mean_PNN'] = np.nan  # SVR Predictions
                df_normal['Pred_mean_Deepesembles'] = np.nan  # KNN Predictions

                df_normal['Pred_std_GPR'] = np.nan  # Linear Regression Predictions
                df_normal['Pred_std_RFCI'] = np.nan  # Random Forest Predictions
                df_normal['Pred_std_PNN'] = np.nan  # SVR Predictions
                df_normal['Pred_std_Deepesembles'] = np.nan  # KNN Predictions

            if self.con_hypopt == True:
                models = ML_Models(x_train, y_train, x_val, y_val, x_test, y_test_, self.seed, best_params_det=best_params_det, best_params_prob=best_params_prob, con_optpara=False, con_deter_models=self.con_deter_models, con_prob_models=self.con_probmodels)
                if self.con_deter_models == True:
                   best_params_det = models.hypopt_determinstic_models()

                if self.con_probmodels == True:
                    best_params_prob = models.hypopt_probablistic_models()                

            # Cross-validation loop
            for i in range(value):
                print(f"\nFold {i + 1}/{value}")
                X_train = x_train[i, :, :].unsqueeze(0)
                Y_train = y_train[i, :].unsqueeze(0)
                X_val = x_val[i, :, :].unsqueeze(0)
                Y_val = y_val[i, :].unsqueeze(0)
                X_test = x_test[i, :, :].unsqueeze(0)
                Y_test = y_test_[i, :]
                y_test += list(Y_test)
                test_idx = test_indices[i]

                models = ML_Models(X_train, Y_train, X_val, Y_val, X_test, Y_test, self.seed, best_params_det=best_params_det, best_params_prob=best_params_prob, con_optpara=self.con_hypopt, con_deter_models=self.con_deter_models, con_prob_models=self.con_probmodels)
                
                if self.con_deter_models == True:
                    models.fit_determinstic_models()
                
                if self.con_probmodels == True:
                    models.fit_probablistic_models()
                    
                if self.con_deter_models == True:
                    predictions_deterministic = models.predict_determinstic_models()

                    # Map the predictions to the corresponding indices in df_normal
                    df_normal.loc[test_idx, 'Pred_LR'] = scaler.inverse_transform(predictions_deterministic['LR'].reshape(-1,1))
                    df_normal.loc[test_idx, 'Pred_RF'] = scaler.inverse_transform(predictions_deterministic['RF'].reshape(-1,1))
                    df_normal.loc[test_idx, 'Pred_SVR'] = scaler.inverse_transform(predictions_deterministic['SVR'].reshape(-1,1))
                    df_normal.loc[test_idx, 'Pred_KNN'] = scaler.inverse_transform(predictions_deterministic['KNN'].reshape(-1,1))
                    df_normal.loc[test_idx, 'Pred_NN'] = scaler.inverse_transform(predictions_deterministic['NN'].reshape(-1,1))

                    allpred_lr += list(predictions_deterministic['LR'])
                    allpred_randomforest += list(predictions_deterministic['RF'])
                    allpred_svr += list(predictions_deterministic['SVR'])
                    allpred_knn += list(predictions_deterministic['KNN'])
                    allpred_neuralnet += list(predictions_deterministic['NN'])

                    evaluation = ErrorMatrix(predictions_deterministic, Y_test, scaler_target, scale_method, con_deter_models=True, con_prob_models=False)
                    #scaled_error = evaluation.evaluate(condition_descale = False)
                    descaled_error = evaluation.evaluate(condition_descale = True)

                    mae = descaled_error['MAE']
                    randomforest_mae.append(mae['RF']), svr_mae.append(mae['SVR']), knn_mae.append(mae['KNN']), neuralnet_mae.append(mae['NN'])

                if self.con_probmodels == True:
                    predictions_probablistic = models.predict_probablistic_models()
                    
                    allpred_gpr = predictions_probablistic['GPR']
                    allpred_rfci = predictions_probablistic['RFCI']
                    allpred_pnn = predictions_probablistic['PNN']
                    allpred_denn = predictions_probablistic['DeepEnsembles']


                    mean_gpr += list(allpred_gpr[0])
                    mean_rfci += list(allpred_rfci[0])
                    mean_pnn += list(allpred_pnn[0])
                    mean_denn += list(allpred_denn[0])

                    
                    std_gpr += list(allpred_gpr[1])
                    std_rfci += list(allpred_rfci[1])
                    std_pnn += list(allpred_pnn[1])
                    std_denn += list(allpred_denn[1])

                    df_normal.loc[test_idx, 'Pred_mean_GPR'] = scaler.inverse_transform(allpred_gpr[0].reshape(-1,1))
                    df_normal.loc[test_idx, 'Pred_std_GPR'] = allpred_gpr[1] * scaler.scale_
                    df_normal.loc[test_idx, 'Pred_mean_RFCI'] = scaler.inverse_transform(allpred_rfci[0].reshape(-1,1))
                    df_normal.loc[test_idx, 'Pred_std_RFCI'] = allpred_rfci[1] * scaler.scale_
                    df_normal.loc[test_idx, 'Pred_mean_PNN'] = scaler.inverse_transform(allpred_pnn[0].reshape(-1,1))
                    df_normal.loc[test_idx, 'Pred_std_PNN'] = allpred_pnn[1] * scaler.scale_
                    df_normal.loc[test_idx, 'Pred_mean_Deepesembles'] = scaler.inverse_transform(allpred_denn[0].reshape(-1,1))
                    df_normal.loc[test_idx, 'Pred_std_Deepesembles'] = allpred_denn[1] * scaler.scale_

                    evaluation = ErrorMatrix(predictions_probablistic, Y_test, scaler_target, scale_method, con_deter_models=False, con_prob_models=True)
                    #scaled_error = evaluation.evaluate(condition_descale = False)
                    descaled_error = evaluation.evaluate(condition_descale = True)

            if self.con_deter_models == True:  
                allfold_predictions = {'LR':np.array(allpred_lr), 'RF':np.array(allpred_randomforest), 'SVR':np.array(allpred_svr), 'KNN':np.array(allpred_knn), 'NN':np.array(allpred_neuralnet)}
                plot = InferencePlots(allfold_predictions, np.array(y_test), scaler_target, scale_method, self.target_label, con_deter_models=True, con_prob_models=False, con_descale=self.con_descale)
                plot.scatterplot()
                plot.errorplot()
                plot.det_model_metrix()

            if self.con_probmodels == True:
                allfold_predictions = {'GPR':(np.array(mean_gpr), np.array(std_gpr)), 'RFCI':(np.array(mean_rfci), np.array(std_rfci)), 'PNN':(np.array(mean_pnn), np.array(std_pnn)), 'DeepEnsembles':(np.array(mean_denn), np.array(std_denn))}
                plot = InferencePlots(allfold_predictions, np.array(y_test), scaler_target, scale_method, self.target_label, con_deter_models=False, con_prob_models=True, con_descale=self.con_descale)
                plot.scatterplot()
                plot.errorplot()
                plot.prob_model_metrix()

        else:
            X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler_target, scale_method, df_normal  = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8]
            best_params_det = {'RF': None, 'SVR':None, 'KNN':None, 'NN':None}
            best_params_prob = {'GPR':None, 'RFCI':None, 'PNN':None, 'DeepEnsembles':None}

            if self.con_hypopt == True:
                models = ML_Models(X_train, Y_train, X_val, Y_val, X_test, Y_test, self.seed, best_params_det=best_params_det, best_params_prob=best_params_prob, con_optpara=False, con_deter_models=self.con_deter_models, con_prob_models=self.con_probmodels)
                if self.con_deter_models == True:
                   best_params_det = models.hypopt_determinstic_models()

                if self.con_probmodels == True:
                    best_params_prob = models.hypopt_probablistic_models()
                
            models = ML_Models(X_train, Y_train, X_val, Y_val, X_test, Y_test, self.seed, best_params_det=best_params_det, best_params_prob=best_params_prob, con_optpara=self.con_hypopt, con_deter_models=self.con_deter_models, con_prob_models=self.con_probmodels)

            if self.con_deter_models == True:
                models.fit_determinstic_models()
                
            if self.con_probmodels == True:
                models.fit_probablistic_models()

            if self.con_deter_models == True:
                prediction_scaled = models.predict_determinstic_models()

                evaluation = ErrorMatrix(prediction_scaled, Y_test, scaler_target, scale_method, con_deter_models=True, con_prob_models=False)
                #scaled_error = evaluation.evaluate(condition_descale = False)
                descaled_error = evaluation.evaluate(condition_descale = True)

                plot = InferencePlots(prediction_scaled, Y_test, scaler_target, scale_method, self.target_label, con_deter_models=True, con_prob_models=False, con_descale=self.con_descale)
                plot.scatterplot()
                plot.errorplot()
                plot.det_model_metrix()

            if self.con_probmodels == True:
                prediction_scaled = models.predict_probablistic_models()
                evaluation = ErrorMatrix(prediction_scaled, Y_test, scaler_target, scale_method, con_deter_models=False, con_prob_models=True)
                #scaled_error = evaluation.evaluate(condition_descale = False)
                descaled_error = evaluation.evaluate(condition_descale = True)

                plot = InferencePlots(prediction_scaled, Y_test, scaler_target, scale_method, self.target_label, con_deter_models=False, con_prob_models=True, con_descale=self.con_descale)
                plot.scatterplot()
                plot.errorplot()
                plot.prob_model_metrix()
        
        return models, best_params_det, best_params_prob, df_normal
    
class ML_Models_Train_transferlearning(BaseEstimator, TransformerMixin):
    def __init__(self, model, method, target_label, custom_split, seed, con_hypopt, con_deter_models, con_prob_models, con_descale):
        self.model = model
        self.method = method
        self.target_label = target_label
        self.custom_split = custom_split
        self.seed = seed
        self.con_hypopt = con_hypopt
        self.con_deter_models = con_deter_models
        self.con_probmodels = con_prob_models
        self.con_descale = con_descale

    def fit(self, inputs):
        return self
        
    def transform(self, inputs):
        key, value = list(self.method.items())[0]

        if (key == 'Kfold' or key == 'Custom-Kfold') and self.custom_split == True:
            x_train, y_train, x_val, y_val, x_test, y_test_, test_indices, scaler_target, scale_method, df_normal = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9]

            if scale_method == 'standardization':
                scaler = scaler_target
            elif scale_method == 'normalization':
                scaler = MinMaxScaler(scaler_target)

            allpred_neuralnet, y_test = [], []
            mean_pnn, std_pnn = [], []

            if self.con_deter_models == True:
                # Initialize the new columns in the dataframe
                df_normal['Pred_NN'] = np.nan  # Neural Net Predictions

            if self.con_probmodels == True:
                df_normal['Pred_mean_PNN'] = np.nan  # SVR Predictions
                df_normal['Pred_std_PNN'] = np.nan  # SVR Predictions
                             

            # Cross-validation loop
            for i in range(value):
                print(f"\nFold {i + 1}/{value}")
                X_train = x_train[i, :, :]
                Y_train = y_train[i, :]
                X_val = x_val[i, :, :]
                Y_val = y_val[i, :]
                X_test = x_test[i, :, :]
                Y_test = y_test_[i, :]
                y_test += list(Y_test)
                test_idx = test_indices[i]

                if self.con_deter_models == True:
                    
                    net = TransferLearningModel(pretrained_model=self.model, con_probmodel=False, lr=0.001, epochs=1000, early_stopping=True, patience=200)
                    net.fit(X_train, Y_train, X_val, Y_val)

                    pred = net.predict(X_test).detach().numpy()
                    df_normal.loc[test_idx, 'Pred_NN'] = scaler.inverse_transform(pred)
                    allpred_neuralnet += list(pred)
                
                if self.con_probmodels == True:
                    net = TransferLearningModel(pretrained_model=self.model, con_probmodel=True, lr=0.001, epochs=1000, early_stopping=True, patience=200)
                    net.fit(X_train, Y_train, X_val, Y_val)
                    
                    predictions = net.predict(X_test)
                    mean_pnn += list(predictions[0])
                    std_pnn += list(predictions[1])

                    df_normal.loc[test_idx, 'Pred_mean_PNN'] = scaler.inverse_transform(predictions[0])
                    df_normal.loc[test_idx, 'Pred_std_PNN'] = predictions[1] * scaler.scale_

            if self.con_deter_models == True:  
                allfold_predictions = {'NN':np.array(allpred_neuralnet)}
                plot = InferencePlots(allfold_predictions, np.array(y_test), scaler_target, scale_method, self.target_label, con_deter_models=True, con_prob_models=False, con_descale=self.con_descale)
                plot.scatterplot()
                plot.errorplot()
                plot.det_model_metrix()

            if self.con_probmodels == True:
                allfold_predictions = {'PNN':(np.array(mean_pnn), np.array(std_pnn))}
                plot = InferencePlots(allfold_predictions, np.array(y_test), scaler_target, scale_method, self.target_label, con_deter_models=False, con_prob_models=True, con_descale=self.con_descale)
                plot.scatterplot()
                plot.errorplot()
                plot.prob_model_metrix()
        else:
            X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler_target, scale_method, df_normal = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8]
            best_params_det = {'RF': None, 'SVR':None, 'KNN':None, 'NN':None}
            best_params_prob = {'GPR':None, 'RFCI':None, 'PNN':None, 'DeepEnsembles':None}

            if self.con_hypopt == True:
                models = ML_Models(X_train, Y_train, X_val, Y_val, X_test, Y_test, self.seed, best_params_det=best_params_det, best_params_prob=best_params_prob, con_optpara=False, con_deter_models=self.con_deter_models, con_prob_models=self.con_probmodels)
                if self.con_deter_models == True:
                   best_params_det = models.hypopt_determinstic_models()

                if self.con_probmodels == True:
                    best_params_prob = models.hypopt_probablistic_models()
                
            models = ML_Models(X_train, Y_train, X_val, Y_val, X_test, Y_test, self.seed, best_params_det=best_params_det, best_params_prob=best_params_prob, con_optpara=self.con_hypopt, con_deter_models=self.con_deter_models, con_prob_models=self.con_probmodels)

            if self.con_deter_models == True:
                models.fit_determinstic_models()
                
            if self.con_probmodels == True:
                models.fit_probablistic_models()

            if self.con_deter_models == True:
                prediction_scaled = models.predict_determinstic_models()

                evaluation = ErrorMatrix(prediction_scaled, Y_test, scaler_target, scale_method, con_deter_models=True, con_prob_models=False)
                #scaled_error = evaluation.evaluate(condition_descale = False)
                descaled_error = evaluation.evaluate(condition_descale = True)

                plot = InferencePlots(prediction_scaled, Y_test, scaler_target, scale_method, self.target_label, con_deter_models=True, con_prob_models=False, con_descale=self.con_descale)
                plot.scatterplot()
                plot.errorplot()
                plot.det_model_metrix()

            if self.con_probmodels == True:
                prediction_scaled = models.predict_probablistic_models()
                evaluation = ErrorMatrix(prediction_scaled, Y_test, scaler_target, scale_method, con_deter_models=False, con_prob_models=True)
                #scaled_error = evaluation.evaluate(condition_descale = False)
                descaled_error = evaluation.evaluate(condition_descale = True)

                plot = InferencePlots(prediction_scaled, Y_test, scaler_target, scale_method, self.target_label, con_deter_models=False, con_prob_models=True, con_descale=self.con_descale)
                plot.scatterplot()
                plot.errorplot()
                plot.prob_model_metrix()
        
        return df_normal