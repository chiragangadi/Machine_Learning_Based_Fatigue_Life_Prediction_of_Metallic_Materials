from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
from hyperopt import fmin, tpe, hp, Trials, space_eval
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, DotProduct
import forestci as fci
from scipy.optimize import fmin_l_bfgs_b
from .CustomFunctions import FeatureDotProduct
from .Datavisualization import plot_hypopt_loss_scatter

class ANN_Arch(nn.Module):
    """
    A fully connected Artificial Neural Network (ANN) architecture with customizable layers, activation functions,
    dropout regularization, batch normalization, and support for conditional probability modeling.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        hidden_dims (list of int): A list of integers specifying the number of neurons in each hidden layer.
        dropout_rate (float): The dropout rate for regularization (between 0 and 1).
        activation_fn (callable): The activation function to be used in hidden layers (e.g., nn.ReLU, nn.LeakyReLU).
        use_batch_norm (bool): Whether to include batch normalization after each hidden layer.
        seed (int, optional): A seed for random number generation to ensure reproducibility. Default is None.
        con_probmodel (bool): A flag to indicate if the model should output conditional probability parameters (mean and variance) for probabilistic predictions.

    Attributes:
        con_probmodel (bool): The flag indicating if conditional probability modeling is enabled.
        seed (int or None): The seed value used for random number generation, or None if not provided.
        output_dim (int): The output dimension of the network (either `output_dim` or `2*output_dim` depending on `con_probmodel`).
        layers (nn.Sequential): The sequential container holding the network layers.
        sigmoid (nn.Sigmoid): The sigmoid activation function (for probabilistic output models).
        softplus (nn.Softplus): The softplus activation function (for variance modeling).
        
    Methods:
        forward(x):
            Performs the forward pass through the network. Returns the model output, either raw logits or probabilistic parameters
            (mean and variance) based on `con_probmodel`.

        _initialize_weights():
            Initializes the weights of the linear layers using Xavier uniform initialization, and biases to zero. Sets the seed 
            for random number generation if specified.

    Notes:
        - If `con_probmodel` is True, the network outputs two values per output dimension: the mean and the variance. The variance is
          passed through a Softplus activation to ensure positivity.
        - If `con_probmodel` is False, the network outputs a single value per output dimension (e.g., logits).
    """

    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate, activation_fn, use_batch_norm, seed, con_probmodel):
        super(ANN_Arch, self).__init__()
        self.con_probmodel = con_probmodel
        self.seed = seed
        if con_probmodel == True:
            self.output_dim = 2*output_dim
        else:
            self.output_dim = output_dim

        # Store layers in a list
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
         
        # Output layer (no dropout or batch norm)
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        
        # Store layers as a ModuleList to use in forward pass
        self.layers = nn.Sequential(*layers) 

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # Automatically initialize weights
        self._initialize_weights()

    def forward(self, x):
        if self.con_probmodel == True:
            out = self.layers(x)
            mean = out[:, 0].unsqueeze(-1)
            var = self.softplus(out[:, 1]).unsqueeze(-1)
            return mean, var
        else:
            out = self.layers(x)
            x = out
            return x
    
    def _initialize_weights(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)  # Set the seed for reproducibility

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)            # Initialize biases to zero


class ANN:
    """
    A class for building, training, and evaluating an artificial neural network (ANN) with optional support for a 
    conditional probability model. The network supports various configurations including dropout, batch normalization,
    and early stopping to improve training performance.

    Attributes:
    ----------
    model : ANN_Arch
        The neural network architecture (model) that defines the layers and structure of the network.
    optimiser : torch.optim.Adam
        The Adam optimizer used to update the model parameters during training.
    criterion : torch.nn.MSELoss
        The loss function used to compute the error between predictions and ground truth.
    con_probmodel : bool
        A flag indicating whether the model is a conditional probability model (True) or not (False).
    epochs : int
        The number of epochs to train the model.
    early_stopping : bool
        A flag to enable early stopping during training to prevent overfitting.
    patience : int
        The number of epochs without improvement before early stopping is triggered.

    Methods:
    -------
    __init__(input_dim, output_dim, hidden_dims, dropout_rate, activation_fn, use_batch_norm, seed, con_probmodel, lr, epochs, early_stopping, patience)
        Initializes the neural network model with the provided parameters.
    fit(X_train, Y_train, X_val, Y_val)
        Trains the model on the given training data (X_train, Y_train) and validates it on (X_val, Y_val).
        Supports early stopping if enabled.
    predict(X_test)
        Makes predictions on the test data (X_test). Returns either the mean and standard deviation of predictions 
        for a conditional probability model, or the direct predictions if not.
    """

    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate, activation_fn, use_batch_norm, seed, con_probmodel, lr, epochs, early_stopping, patience):
        self.model = ANN_Arch(input_dim, output_dim, hidden_dims, dropout_rate, activation_fn, use_batch_norm, seed, con_probmodel)
        self.optimiser = optim.Adam(self.model.parameters(), lr)
        self.criterion = nn.MSELoss()
        self.con_probmodel = con_probmodel
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience

    def fit(self, X_train, Y_train, X_val, Y_val):
        Y_train = Y_train.unsqueeze(-1)
        Y_val = Y_val.unsqueeze(-1)
        train_runningloss = []
        val_runningloss = []
        best_loss = float('inf')
        wait = 0

        for epoch in range(self.epochs):
            self.model.train()

            if self.con_probmodel == True:
                mean, var = self.model(X_train)
                train_loss = F.gaussian_nll_loss(mean, Y_train, var)
            else:
                predictions = self.model(X_train)
                train_loss = self.criterion(predictions, Y_train)
    
            train_runningloss.append(train_loss.item())
            
            self.optimiser.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            self.model.eval()
            with torch.no_grad():
                if self.con_probmodel == True:
                    mean, var = self.model(X_val)
                    val_loss = F.gaussian_nll_loss(mean, Y_val, var)
                else:
                    predictions = self.model(X_val)
                    val_loss = self.criterion(predictions, Y_val)

                val_runningloss.append(val_loss.item())

            #print(f"Epoch{epoch+1}:  Training loss: {train_loss}   Validation loss: {val_loss}")

            # Check if early stopping is enabled
            if self.early_stopping == True:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_weights = copy.deepcopy(self.model.state_dict())  # Save the best weights
                    wait = 0  # Reset patience counter
                else:
                    wait += 1  # Increment patience counter
                    #print(f'No improvement, patience counter: {wait}')

                # Check if patience limit has been reached
                if wait >= self.patience:
                    #print('Early stopping triggered')
                    self.model.load_state_dict(best_weights)  # Restore model weights to the best observed state
                    break
        
        # plt.plot(train_runningloss, label='Train loss')
        # plt.plot(val_runningloss, label='Validation loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('MSE loss')
        # #plt.title('Train Validation Losses')
        # plt.legend()
        # plt.show()
    
    def predict(self, X_test):
        self.model.eval()

        if self.con_probmodel == True:
            mean, var = self.model(X_test)
            mean = mean.detach().numpy().ravel()
            std = np.sqrt(var.detach().numpy().ravel())
            predictions = (mean, std)
        else:
            predictions = self.model(X_test)

        return predictions


class ANN_Regressor:
    """
    A class for building, training, hyperparameter tuning, and predicting using an Artificial Neural Network (ANN) regressor.

    Attributes:
    ----------
    input_dim : int
        The number of input features for the ANN.
    output_dim : int
        The number of output features for the ANN.
    seed : int
        The random seed for model initialization.
    con_probmodel : bool
        Flag indicating whether to use a probabilistic neural network model.
    neuralnet : object
        An instance of the ANN class.
    
    Methods:
    --------
    __init__(self, input_dim, output_dim, seed, best_params, con_optpara, con_probmodel, epochs, early_stopping, patience):
        Initializes the ANN_Regressor with the specified parameters, either creating a standard ANN or a probabilistic neural network (PNN).
    
    train(self, X_train, Y_train, X_val, Y_val):
        Trains the neural network using the provided training and validation data.

    optimize_hyperparameters(self, X_train, Y_train, X_val, Y_val):
        Optimizes the hyperparameters of the ANN using the given training and validation data. This method employs a hyperparameter optimization algorithm (Hyperopt) and returns the best hyperparameters.

    predict(self, X_test):
        Makes predictions on the provided test data using the trained neural network.

    """
    def __init__(self, input_dim, output_dim, seed, best_params, con_optpara, con_probmodel, epochs, early_stopping, patience):
        if con_optpara == True:
            self.neuralnet = ANN(input_dim, output_dim, best_params['Hidden layer'], float(best_params['Dropout_rate']), 
                                best_params['Activation function'], best_params['Batch_norm'], seed, con_probmodel, float(best_params['lr']), epochs=epochs, early_stopping=early_stopping, patience=patience)
        elif con_optpara == False:
            self.neuralnet = ANN(input_dim=input_dim, output_dim=output_dim, hidden_dims=[128,256,128], dropout_rate=0.4, activation_fn=nn.Sigmoid, use_batch_norm=True, 
                                seed=seed, con_probmodel=con_probmodel, lr=0.001, epochs=epochs, early_stopping=early_stopping, patience=patience)
        else:
            print('Model is not initialised, reinitialise the model with right parameters')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self.con_probmodel = con_probmodel

    def train(self, X_train, Y_train, X_val, Y_val):
        self.neuralnet.fit(X_train[0,:,:], Y_train[0,:], X_val[0,:,:], Y_val[0,:])
        if self.con_probmodel == True:
            print('Training of Probabilistic Neural network completed')
        else:
            print('Training of Neural network completed')

    def optimize_hyperparameters(self, X_train, Y_train, X_val, Y_val):
        loss_list = []
        def objective_nn(params):
            total_loss = 0
            for i in range(X_train.shape[0]):  
                model = ANN(self.input_dim, self.output_dim, params['hidden_layers'], float(params['dropout_rate']), 
                        params['activation_fn'], params['use_batch_norm'], self.seed, self.con_probmodel, float(params['lr']), epochs=2000, 
                        early_stopping=True, patience=100)
                model.fit(X_train[i,:,:], Y_train[i,:], X_val[i,:,:], Y_val[i,:])
                if self.con_probmodel == True:
                    mean_std = model.predict(X_val[i,:,:])
                    mean = torch.tensor(mean_std[0])
                    var = torch.square(torch.tensor(mean_std[1]))

                    loss = F.gaussian_nll_loss(mean, Y_val[i,:], var)
                    total_loss += loss.item()
                
                else:
                    predictions = model.predict(X_val[i,:,:]).detach().numpy()
                    mse = mean_squared_error(Y_val[i,:], predictions)
                    total_loss += mse.item()
            loss_list.append(total_loss)

            return total_loss

        search_space_nn = {
            'hidden_layers': hp.choice('hidden_layers', [[hp.choice(f'layer_{i}_neurons_{num_layers}', [32, 64, 128, 256]) for i in range(num_layers)] for num_layers in range(1, 5)]),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
            'activation_fn': hp.choice('activation_fn', [nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.ELU, nn.Tanh, nn.SELU]),
            'use_batch_norm': hp.choice('use_batch_norm', [True, False]),
            'lr': hp.loguniform('learning_rate', -5, -2),  # Values in range ~[1e-5, 1e-2]
        }

        trials_nn = Trials()
        best_params_nn = fmin(fn=objective_nn, space=search_space_nn, algo=tpe.suggest, max_evals=50, trials=trials_nn)

        # Map indices to activation functions
        activation_fn_mapping = {
            0: nn.ReLU,
            1: nn.Sigmoid,
            2: nn.LeakyReLU,
            3: nn.ELU,
            4: nn.Tanh,
            5: nn.SELU
        }

        activation_fn = activation_fn_mapping[best_params_nn['activation_fn']]

       # Evaluate the full space for hidden layers using the returned best index
        evaluated_space = space_eval(search_space_nn, best_params_nn)

        # Extract neurons list
        neurons_list = evaluated_space['hidden_layers']  # This will now be the fully resolved list of neurons


        # Final parameter dictionary
        best_params = {
            'Hidden layer': neurons_list,
            'Dropout_rate': best_params_nn['dropout_rate'],
            'Activation function': activation_fn,
            'Batch_norm': best_params_nn['use_batch_norm'],
            'lr': best_params_nn['learning_rate']
        }

        if self.con_probmodel == True:
            print("Best hyperparameters for Probabilstic Neural Network:", best_params)
            plot_hypopt_loss_scatter(loss_list, 'PNN', 'NLL')
        else:
            print("Best hyperparameters for Neural Network:", best_params)
            plot_hypopt_loss_scatter(loss_list, 'NN', 'MSE')

        return best_params

    def predict(self, X_test):
        return self.neuralnet.predict(X_test[0,:,:])


class RandomForest_Regressor:
    """
    A Random Forest Regressor model used for predicting continuous target values.

    Attributes:
        randomforest (RandomForestRegressor): An instance of the RandomForestRegressor model from sklearn.
    
    Methods:
        __init__(best_params, con_optpara): Initializes the RandomForest_Regressor model with either custom or default hyperparameters.
        train(X_train, Y_train): Trains the model using the provided training data (X_train, Y_train).
        optimize_hyperparameters(X_train, Y_train, X_val, Y_val): Optimizes the hyperparameters of the model using Hyperopt for best performance.
        predict(X_test): Predicts the target values using the trained model on the provided test data (X_test).
    """

    def __init__(self, best_params, con_optpara):
        """
        Initializes the RandomForest_Regressor with specified hyperparameters.

        Args:
            best_params (dict): Dictionary of hyperparameters for the RandomForestRegressor.
            con_optpara (bool): If True, use the provided best_params for model initialization.
                                 If False, initializes the model with default parameters.

        Raises:
            ValueError: If con_optpara is neither True nor False.
        """
        if con_optpara == True:
            self.randomforest = RandomForestRegressor(n_estimators=int(best_params['n_estimators']), max_depth=int(best_params['max_depth']))
        elif con_optpara == False:
            self.randomforest = RandomForestRegressor(n_estimators=500, max_depth=50)
        else:
            print('Model is not initialised, reinitialise the model with right parameters')

    def train(self, X_train, Y_train):
        self.randomforest.fit(X_train[0,:,:], Y_train[0,:])
        print('Training of Random forest model completed')

    def optimize_hyperparameters(self, X_train, Y_train, X_val, Y_val):
        loss_list = []
        def objective_rf(params):
            total_loss = 0
            for i in range(X_train.shape[0]):
                model = RandomForestRegressor(
                    n_estimators=int(params['n_estimators']), 
                    max_depth=int(params['max_depth']),
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features']
                )
                model.fit(X_train[i, :, :], Y_train[i, :])
                predictions = model.predict(X_val[i, :, :])
                mse = mean_squared_error(Y_val[i, :], predictions)
                total_loss += mse
            loss_list.append(total_loss.item())
            return total_loss

        search_space_rf = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),  # Number of trees in the forest
            'max_depth': hp.quniform('max_depth', 10, 200, 20),         # Maximum depth of the tree
            'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.5),  # Minimum fraction of samples required to split an internal node
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.5),   # Minimum fraction of samples required to be at a leaf node
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None])  # Number of features to consider at each split
        }

        trials_rf = Trials()
        best_params = fmin(fn=objective_rf, space=search_space_rf, algo=tpe.suggest, max_evals=50, trials=trials_rf)

        print("Best hyperparameters for Random Forest:", best_params)
        plot_hypopt_loss_scatter(loss_list, 'RF', 'MSE')

        return best_params

    def predict(self, X_test):
        return self.randomforest.predict(X_test[0,:,:])


class SupportVector_Regressor:
    """
    A Support Vector Regressor (SVR) model for regression tasks. The class provides methods to initialize the model, 
    train it on data, optimize hyperparameters using Bayesian optimization, and make predictions.

    Attributes:
        svr (svm.SVR): The SVR model object initialized with specified hyperparameters.
    
    Methods:
        __init__(self, best_params, con_optpara):
            Initializes the SVR model with either optimized or default hyperparameters.
        
        train(self, X_train, Y_train):
            Trains the SVR model using the provided training data.
        
        optimize_hyperparameters(self, X_train, Y_train, X_val, Y_val):
            Optimizes the SVR hyperparameters (`C` and `epsilon`) using Bayesian optimization and cross-validation.
        
        predict(self, X_test):
            Predicts target values for the given test data using the trained SVR model.
    """
    def __init__(self, best_params, con_optpara):
        """
        Initializes the Support Vector Regressor (SVR) model.
        
        Parameters:
            best_params (dict): A dictionary containing the hyperparameters `C` and `epsilon` for the model.
            con_optpara (bool): If True, initializes the model with the given best parameters. 
                                 If False, initializes the model with default values.
        """
        if con_optpara == True:
            self.svr = svm.SVR(C=float(best_params['C']), epsilon=float(best_params['epsilon']))
        elif con_optpara == False:
            self.svr = svm.SVR(C=5, epsilon=0.03)
        else:
            print('Model is not initialised, reinitialise the model with right parameters')
    
    def train(self,X_train, Y_train):
        self.svr.fit(X_train[0,:,:], Y_train[0,:])
        print('Training of SVR model completed')

    def optimize_hyperparameters(self, X_train, Y_train, X_val, Y_val):
        loss_list = []
        def objective_svr(params):
            total_loss = 0
            for i in range(X_train.shape[0]):
                model = svm.SVR(C=float(params['C']), epsilon=float(params['epsilon']))
                model.fit(X_train[i,:,:], Y_train[i,:])
                predictions = model.predict(X_val[i,:,:])
                mse = mean_squared_error(Y_val[i,:], predictions)
                total_loss += mse
            loss_list.append(total_loss.item())
            return total_loss
        
        search_space_svr = {
            'C': hp.loguniform('C', np.log(0.01), np.log(100)),
            'epsilon': hp.uniform('epsilon', 0.01, 0.1)
        }

        trials_svr = Trials()
        best_params = fmin(fn=objective_svr, space=search_space_svr, algo=tpe.suggest, max_evals=50, trials=trials_svr)
        print("Best hyperparameters for SVR:", best_params)
        plot_hypopt_loss_scatter(loss_list, 'SVR', 'MSE')

        return best_params

    def predict(self, X_test):
        return self.svr.predict(X_test[0,:,:])


class KNearestNeighbor_Regressor:
    """
    A K-Nearest Neighbors Regressor model for predicting continuous values.
    
    This class uses KNN for regression and allows hyperparameter optimization using 
    the Hyperopt library. It supports model training, hyperparameter optimization, 
    and prediction on new data.

    Attributes:
    -----------
    knn : KNeighborsRegressor
        An instance of the scikit-learn KNeighborsRegressor model.
    
    Methods:
    --------
    __init__(best_params, con_optpara):
        Initializes the KNN regressor with either default or best hyperparameters.
    
    train(X_train, Y_train):
        Trains the KNN model using the provided training data.
    
    optimize_hyperparameters(X_train, Y_train, X_val, Y_val):
        Optimizes the hyperparameters of the KNN model using Hyperopt.
    
    predict(X_test):
        Predicts the output for the given test data using the trained KNN model.
    """
    def __init__(self, best_params, con_optpara):
        """
        Initializes the KNN regressor model.
        
        Parameters:
        -----------
        best_params : dict
            Dictionary containing the best hyperparameters for the model.
        
        con_optpara : bool
            If True, the model is initialized with optimized hyperparameters from 'best_params'.
            If False, the model is initialized with default hyperparameters.
        """
        if con_optpara == True:
            self.knn = neighbors.KNeighborsRegressor(n_neighbors=int(best_params['n_neighbors']))
        elif con_optpara == False:
            self.knn = neighbors.KNeighborsRegressor(n_neighbors=5)
        else:
            print('Model is not initialised, reinitialise the model with right parameters')
    
    def train(self,X_train, Y_train):
        self.knn.fit(X_train[0,:,:], Y_train[0,:])
        print('Training of KNN model completed')

    def optimize_hyperparameters(self, X_train, Y_train, X_val, Y_val):
        loss_list = []
        def objective_knn(params):
            total_loss = 0
            for i in range(X_train.shape[0]):
                model = neighbors.KNeighborsRegressor(n_neighbors=int(params['n_neighbors']))
                model.fit(X_train[i,:,:], Y_train[i,:])
                predictions = model.predict(X_val[i,:,:])
                mse = mean_squared_error(Y_val[i,:], predictions)
                total_loss += mse
            loss_list.append(total_loss.item())
            return total_loss

        search_space_knn = {
            'n_neighbors': hp.quniform('n_neighbors', 2, 30, 1)
        }

        trials_knn = Trials()
        best_params = fmin(fn=objective_knn, space=search_space_knn, algo=tpe.suggest, max_evals=50, trials=trials_knn)
        print("Best hyperparameters for KNN:", best_params)
        plot_hypopt_loss_scatter(loss_list, 'KNN', 'MSE')

        return best_params

    def predict(self, X_test):
        return self.knn.predict(X_test[0,:,:])


class GaussianProcess_Regressor:
    """
    A class for performing Gaussian Process Regression with hyperparameter optimization.

    This class allows you to initialize and train a Gaussian Process Regressor (GPR) using
    different kernels and optimize hyperparameters for the model using the `hyperopt` package.
    
    Attributes:
    - length_scale_bounds (tuple): Bounds for the length scale parameter of kernels.
    - sigma_0_bounds (tuple): Bounds for the sigma_0 parameter of kernels.
    - gaussian_process_regressor (GaussianProcessRegressor): The underlying scikit-learn GPR model.
    
    Methods:
    - train(X_train, Y_train): Fit the GPR model to the training data.
    - optimize_hyperparameters(X_train, Y_train, X_val, Y_val, max_evals): Optimize hyperparameters
      for the GPR model using the `hyperopt` library.
    - predict(X_test, return_std=True): Make probabilistic predictions using the trained GPR model.
    """

    def __init__(self, num_features, best_params, con_optpara):
        """
        Initialize the Gaussian Process Regressor.

        Parameters:
        - num_features (int): The number of input features for the kernel.
        - best_params (dict): A dictionary containing the best hyperparameters for the model.
        - con_optpara (bool): If True, use the `best_params` to initialize the model with the GaussianProcessRegressor;
                               if False, use default kernel configurations.
        
        Initializes the Gaussian Process Regressor with either the provided hyperparameters or default kernels.
        If `con_optpara` is True, it initializes the model using the given `best_params` dictionary.
        Otherwise, it defaults to a combination of RBF and Linear or Matern kernels with a constant kernel.
        """
        self.length_scale_bounds = (1e-100, 1e100)
        self.sigma_0_bounds = (1e-100, 1e100)
        if con_optpara == True:
            self.gaussian_process_regressor = GaussianProcessRegressor(kernel=best_params['kernel'], alpha=best_params['alpha'], n_restarts_optimizer=5, normalize_y=False)
        elif con_optpara == False:
            # Default kernel: constant kernel * RBF kernel
            kernel = C(1.0, (1e-3, 1e3)) * (RBF(length_scale=np.ones(num_features), length_scale_bounds=self.length_scale_bounds) + DotProduct(sigma_0=1, sigma_0_bounds=(1e-5, 1e5)))
            self.gaussian_process_regressor = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=5, normalize_y=False)
        else:
            print('Model is not initialised, reinitialise the model with right parameters')

    def train(self, X_train, Y_train):
        """
        Fit the Gaussian Process Regressor to the training data.
        
        Parameters:
        - X_train: Training features (numpy array or pandas DataFrame)
        - Y_train: Training targets (numpy array or pandas DataFrame)
        """
        self.gaussian_process_regressor.fit(X_train[0,:,:], Y_train[0,:])
        print("Training of Gaussian Process Regressor completed")

    def optimize_hyperparameters(self, X_train, Y_train, X_val, Y_val, max_evals=50):
        """
        Optimize Gaussian Process hyperparameters using hyperopt.
        
        Parameters:
        - X_train: Training features
        - Y_train: Training targets
        - X_val: Validation features
        - Y_val: Validation targets
        - max_evals: Maximum number of evaluations for hyperopt

        Returns:
        - best_params: Dictionary of best hyperparameters found
        """
        loss_list = []
        num_features = X_train.shape[2]  # Determine the number of input features

        def objective(params):
            # Define kernel based on the selected type and hyperparameters
            # Define kernel based on the selected type and hyperparameters
            if params['kernel'] == 'C*RBF':
                kernel = C(constant_value=params['constant']) * RBF(length_scale=params['length_scales_rbf'], length_scale_bounds=self.length_scale_bounds)
            elif params['kernel'] == 'C*Matern':
                kernel = C(constant_value=params['constant']) * Matern(length_scale=params['length_scales_matern'], length_scale_bounds=self.length_scale_bounds, nu=params['nu'])
            elif params['kernel'] == 'C*(RBF+Linear)':
                kernel = C(constant_value=params['constant']) * (RBF(length_scale=params['length_scales_rbf'], length_scale_bounds=self.length_scale_bounds) + DotProduct(sigma_0=params['sigma_0'], sigma_0_bounds=self.sigma_0_bounds))
            elif params['kernel'] == 'C*(Matern+Linear)':
                kernel = C(constant_value=params['constant']) * (Matern(length_scale=params['length_scales_matern'], length_scale_bounds=self.length_scale_bounds, nu=params['nu']) + DotProduct(sigma_0=params['sigma_0'], sigma_0_bounds=self.sigma_0_bounds))
            elif params['kernel'] == 'C*(Matern+RBF+Linear)':
                kernel = C(constant_value=params['constant']) * (Matern(length_scale=params['length_scales_matern'], length_scale_bounds=self.length_scale_bounds, nu=params['nu']) + RBF(length_scale=params['length_scales_rbf'], length_scale_bounds=self.length_scale_bounds) + DotProduct(sigma_0=params['sigma_0'], sigma_0_bounds=self.sigma_0_bounds))
            elif params['kernel'] == 'RBF':
                kernel = RBF(length_scale=params['length_scales_rbf'], length_scale_bounds=self.length_scale_bounds)
            elif params['kernel'] == 'Matern':
                kernel = Matern(length_scale=params['length_scales_matern'], length_scale_bounds=self.length_scale_bounds, nu=params['nu'])
            elif params['kernel'] == 'RBF+Linear':
                kernel = (RBF(length_scale=params['length_scales_rbf'], length_scale_bounds=self.length_scale_bounds) + DotProduct(sigma_0=params['sigma_0'], sigma_0_bounds=self.sigma_0_bounds))
            elif params['kernel'] == 'Matern+Linear':
                kernel = (Matern(length_scale=params['length_scales_matern'], length_scale_bounds=self.length_scale_bounds, nu=params['nu']) + DotProduct(sigma_0=params['sigma_0'], sigma_0_bounds=self.sigma_0_bounds))
            elif params['kernel'] == 'Matern+RBF+Linear':
                kernel = (Matern(length_scale=params['length_scales_matern'], length_scale_bounds=self.length_scale_bounds, nu=params['nu']) + RBF(length_scale=params['length_scales_rbf'], length_scale_bounds=self.length_scale_bounds) + DotProduct(sigma_0=params['sigma_0'], sigma_0_bounds=self.sigma_0_bounds))
            
            total_loss = 0
            for i in range(X_train.shape[0]):
                model = GaussianProcessRegressor(kernel=kernel, alpha=params['alpha'], n_restarts_optimizer=5, normalize_y=False)
                model.fit(X_train[i,:,:], Y_train[i,:])
                mean, std = model.predict(X_val[i,:,:], return_std=True)
                mean = torch.tensor(mean)
                var = torch.tensor(np.square(std))
                loss = F.gaussian_nll_loss(mean, Y_val[i,:], var)
                total_loss += loss.item()
            loss_list.append(total_loss)
            return total_loss

        # Define the search space for hyperparameters
        search_space = {
            'kernel': hp.choice('kernel', ['C*RBF', 'C*Matern', 'C*(RBF+Linear)', 'C*(Matern+Linear)', 'C*(Matern+RBF+Linear)', 'RBF', 'Matern', 'RBF+Linear', 'Matern+Linear', 'Matern+RBF+Linear']),     
            'constant': hp.quniform('constant', 0.5, 1.5, 0.001),
            'length_scales_rbf': [hp.quniform(f'ls_rbf{i}', 1, 2, 0.01) for i in range(num_features)],
            'length_scales_matern': [hp.quniform(f'ls_matern{i}', 1, 2, 0.01) for i in range(num_features)],
            'alpha': hp.quniform('alpha', 1e-4, 1e-2, 1e-4),
            'nu': hp.choice('nu', [0.5, 1.5, 2.5]),                                                             # Only relevant for Matern kernel
            'sigma_0': hp.quniform('sigma_0',0.1, 2, 0.01)                                                      # Only relevant for Linear kernel                                                      
        }

        # Run optimization
        trials = Trials()
        best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        # Decode best_params from hyperopt
        kernel_type = ['C*RBF', 'C*Matern', 'C*(RBF+Linear)', 'C*(Matern+Linear)', 'C*(Matern+RBF+Linear)', 'RBF', 'Matern', 'RBF+Linear', 'Matern+Linear', 'Matern+RBF+Linear'][best_params['kernel']] 
        length_scales_rbf = [best_params[f'ls_rbf{i}'] for i in range(num_features)]
        length_scales_matern = [best_params[f'ls_matern{i}'] for i in range(num_features)]
        sigma_0 = best_params['sigma_0']
        nu = [0.5, 1.5, 2.5]

        if kernel_type == 'C*RBF':
            best_kernel = C(best_params['constant']) * RBF(length_scales_rbf, length_scale_bounds=self.length_scale_bounds)
        elif kernel_type == 'C*Matern':
            best_kernel = C(best_params['constant']) * Matern(length_scales_matern, length_scale_bounds=self.length_scale_bounds, nu=nu[best_params['nu']])
        elif kernel_type == 'C*(RBF+Linear)':
            best_kernel = C(best_params['constant']) * (RBF(length_scales_rbf, length_scale_bounds=self.length_scale_bounds) + DotProduct(sigma_0, sigma_0_bounds=self.sigma_0_bounds))
        elif kernel_type == 'C*(Matern+Linear)':
            best_kernel = C(best_params['constant']) * (Matern(length_scales_matern, length_scale_bounds=self.length_scale_bounds, nu=nu[best_params['nu']]) + DotProduct(sigma_0, sigma_0_bounds=self.sigma_0_bounds))
        elif kernel_type == 'C*(Matern+RBF+Linear)':
            best_kernel = C(best_params['constant']) * (Matern(length_scales_matern, length_scale_bounds=self.length_scale_bounds, nu=nu[best_params['nu']]) + RBF(length_scales_rbf, length_scale_bounds=self.length_scale_bounds) + DotProduct(sigma_0, sigma_0_bounds=self.sigma_0_bounds))
        elif kernel_type == 'RBF':
            best_kernel = RBF(length_scales_rbf, length_scale_bounds=self.length_scale_bounds)
        elif kernel_type == 'Matern':
            best_kernel = Matern(length_scales_matern, length_scale_bounds=self.length_scale_bounds, nu=nu[best_params['nu']])
        elif kernel_type == 'RBF+Linear':
            best_kernel = (RBF(length_scales_rbf, length_scale_bounds=self.length_scale_bounds) + DotProduct(sigma_0, sigma_0_bounds=self.sigma_0_bounds))
        elif kernel_type == 'Matern+Linear':
            best_kernel = (Matern(length_scales_matern, length_scale_bounds=self.length_scale_bounds, nu=nu[best_params['nu']]) + DotProduct(sigma_0, sigma_0_bounds=self.sigma_0_bounds))
        elif kernel_type == 'Matern+RBF+Linear':
            best_kernel = (Matern(length_scales_matern, length_scale_bounds=self.length_scale_bounds, nu=nu[best_params['nu']]) + RBF(length_scales_rbf, length_scale_bounds=self.length_scale_bounds) + DotProduct(sigma_0, sigma_0_bounds=self.sigma_0_bounds))
        
        best_params = {'kernel':best_kernel, 'alpha':best_params['alpha']}
        print("Best hyperparameters for Gaussian Process Regressor:", best_params)
        plot_hypopt_loss_scatter(loss_list, 'GPR', 'NLL')
        
        return best_params

    def predict(self, X_test, return_std=True):
        """
        Make probabilistic predictions using the Gaussian Process Regressor.
        
        Parameters:
        - X_test: Test features (numpy array or pandas DataFrame)
        - return_std: If True, returns standard deviation of predictions
        
        Returns:
        - mean: Mean predictions (numpy array)
        - std: Standard deviation of predictions (numpy array, only if return_std=True)
        """
        mean, std = self.gaussian_process_regressor.predict(X_test[0,:,:], return_std=True)
        if return_std:
            return (mean, std)
        else:
            return mean


class RandomForest_CI:
    """
    A class for training and optimizing a RandomForestRegressor model with an optional jackknife procedure 
    for uncertainty estimation (confidence intervals). The class provides functionality for model training, 
    hyperparameter optimization using Hyperopt, and prediction with error estimation.

    Attributes:
        randomforest (RandomForestRegressor): A trained RandomForestRegressor model.
    
    Methods:
        __init__(best_params, con_optpara):
            Initializes the RandomForest_CI class with the specified parameters and configuration.
        
        train(X_train, Y_train):
            Trains the RandomForest model using the provided training data.
        
        optimize_hyperparameters(X_train, Y_train, X_val, Y_val):
            Optimizes the hyperparameters of the RandomForest model using Hyperopt to minimize the MSE.
        
        predict(X_train, X_test):
            Makes predictions on the test data and estimates the standard deviation of the predictions.
    """
    def __init__(self, best_params, con_optpara):
        """
        Initializes the RandomForest_CI class with the given hyperparameters and configuration option.

        Parameters:
            best_params (dict): A dictionary containing the best hyperparameters for the RandomForest model.
            con_optpara (bool): A flag indicating whether to initialize the RandomForest model with the provided
                                 best hyperparameters (True) or default values (False).

        Raises:
            ValueError: If con_optpara is neither True nor False.
        
        Initializes the RandomForestRegressor model based on the provided parameters.
        If con_optpara is True, the model is initialized using the best_params dictionary.
        If con_optpara is False, the model is initialized with default values (n_estimators=2500, max_depth=50).
        """
        if con_optpara == True:
            self.randomforest = RandomForestRegressor(n_estimators=int(best_params['n_estimators']), max_depth=int(best_params['max_depth']), random_state=42)
        elif con_optpara == False:
            self.randomforest = RandomForestRegressor(n_estimators=2500, max_depth=50, random_state=42)
        else:
            print('Model is not initialised, reinitialise the model with right parameters')
    
    def train(self, X_train, Y_train):
        """
        Train the RandomForest model.
        """
        self.randomforest.fit(X_train[0,:,:], Y_train[0,:])
        print('Training of Random Forest with Jackknife model completed')
    
    def optimize_hyperparameters(self, X_train, Y_train, X_val, Y_val):
        """
        Optimize hyperparameters using Hyperopt.
        """
        loss_list = []
        def objective_rf(params):
            total_loss = 0
            for i in range(X_train.shape[0]):
                # Define the model with current parameters
                model = RandomForestRegressor(
                    n_estimators=int(params['n_estimators']), 
                    max_depth=int(params['max_depth']),
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features']
                )
                model.fit(X_train[i,:,:], Y_train[i,:])
                predictions = model.predict(X_val[i,:,:])
                mse = mean_squared_error(Y_val[i,:], predictions)
                total_loss += mse
            loss_list.append(total_loss.item())
            return total_loss

        # Define the search space for hyperparameters
        search_space_rf = {
            'n_estimators': hp.quniform('n_estimators', 2000, 3000, 5),
            'max_depth': hp.quniform('max_depth', 10, 200, 20),         # Maximum depth of the tree
            'min_samples_split': hp.uniform('min_samples_split', 0.01, 0.5),  # Minimum fraction of samples required to split an internal node
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01, 0.5),   # Minimum fraction of samples required to be at a leaf node
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None])  # Number of features to consider at each split
        }

        trials_rf = Trials()
        best_params = fmin(fn=objective_rf, space=search_space_rf, algo=tpe.suggest, max_evals=50, trials=trials_rf)

        print("Best hyperparameters for Random Forest with Jackknife:", best_params)
        plot_hypopt_loss_scatter(loss_list, 'RFCI', 'MSE')

        return best_params
    
    def predict(self, X_train, X_test):
        mean_prediction = self.randomforest.predict(X_test[0,:,:])
        variance = fci.random_forest_error(self.randomforest, X_train[0,:,:].shape, X_test[0,:,:])
        std_prediction = np.sqrt(variance)
        return (mean_prediction, std_prediction)


class DeepEnsemble_Regressor:
    def __init__(self, num_networks, input_dim, output_dim, seed, best_params, con_optpara, con_probmodel, epochs, early_stopping, patience):
        """
        Initialize the Deep Ensemble with multiple independent neural networks.

        Parameters:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            num_networks (int): Number of networks in the ensemble.
            hidden_dims (list): List of hidden layer sizes for each network.
            dropout_rate (float): Dropout rate for regularization.
            activation_fn (callable): Activation function (e.g., nn.ReLU).
            use_batch_norm (bool): Whether to use batch normalization.
            lr (float): Learning rate for training.
            epochs (int): Number of training epochs.
            early_stopping (bool): Whether to use early stopping.
            patience (int): Patience for early stopping.
        """
        self.num_networks = num_networks
        if con_optpara == True:
            self.networks = [
            ANN(input_dim, output_dim, best_params['Hidden layer'], float(best_params['Dropout_rate']), best_params['Activation function'], best_params['Batch_norm'], (seed+i), con_probmodel, float(best_params['lr']), epochs=epochs, early_stopping=early_stopping, patience=patience)
            for i in range(self.num_networks)
            ]
        elif con_optpara == False:
            self.networks = [
                ANN(input_dim=input_dim, output_dim=output_dim, hidden_dims=[64,64], dropout_rate=0.4, activation_fn=nn.Sigmoid, use_batch_norm=True, seed=(seed+i), con_probmodel=con_probmodel, lr=0.001, epochs=epochs, early_stopping=early_stopping, patience=patience)
                for i in range(num_networks)
            ]
        else:
            print('Model is not initialised, reinitialise the model with right parameters')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.con_probmodel = con_probmodel
        self.seed = seed

    def train(self, X_train, Y_train, X_val, Y_val):
        """
        Train all networks in the ensemble.
        """
        for i, net in enumerate(self.networks):
            print(f"Training Deepensemble network {i + 1}/{self.num_networks}...")
            net.fit(X_train[0,:,:], Y_train[0,:], X_val[0,:,:], Y_val[0,:])
        print("Training of all ensemble networks completed.")

    def optimize_hyperparameters(self, X_train, Y_train, X_val, Y_val):
        def objective_nn(params):
            total_loss = 0
            for j in range(X_train.shape[0]):
                models = [
                    ANN(self.input_dim, self.output_dim, params['hidden_layers'], float(params['dropout_rate']),
                        params['activation_fn'], params['use_batch_norm'], self.seed, self.con_probmodel, 
                        float(params['lr']), epochs=2000, early_stopping=True, patience=100)
                    for _ in range(self.num_networks)
                ]

                mean_allnet = np.zeros((self.num_networks, len(X_val[j, :, :])))
                std_allnet = np.zeros((self.num_networks, len(X_val[j, :, :])))
                mean_pred = np.zeros((1, len(X_val[j, :, :])))
                total_std = np.zeros((1, len(X_val[j, :, :])))
                aleatoric_std = np.zeros((1, len(X_val[j, :, :])))

                for i, net in enumerate(models):
                    net.fit(X_train[j, :, :], Y_train[j, :], X_val[j, :, :], Y_val[j, :])
                    pred = net.predict(X_val[j, :, :])
                    mean = pred[0]
                    std = pred[1]
                    mean_allnet[i, :] = mean
                    std_allnet[i, :] = std

                for i in range(len(X_val[0, :, :])):
                    mean_pred[0, i] = np.mean(mean_allnet[:, i])
                    total_std[0, i] = np.sqrt(np.mean(std_allnet[:, i] ** 2 + mean_allnet[:, i] ** 2) - mean_pred[0, i] ** 2)
                    aleatoric_std[0, i] = np.mean(std_allnet[:, i])

                mean = torch.tensor(mean_pred.ravel())
                var = torch.tensor(np.square(total_std).ravel())

                loss = F.gaussian_nll_loss(mean, Y_val[j, :], var)
                total_loss += loss.item()
            return total_loss

        search_space_nn = {
            'hidden_layers': hp.choice('hidden_layers', [[hp.choice(f'layer_{i}_neurons_{num_layers}', [32, 64, 128, 256]) for i in range(num_layers)] for num_layers in range(1, 5)]),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
            'activation_fn': hp.choice('activation_fn', [nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.ELU, nn.Tanh, nn.SELU]),
            'use_batch_norm': hp.choice('use_batch_norm', [True, False]),
            'lr': hp.loguniform('learning_rate', -5, -2),  # Values in range ~[1e-5, 1e-2]
        }

        trials_nn = Trials()
        best_params_nn = fmin(fn=objective_nn, space=search_space_nn, algo=tpe.suggest, max_evals=50, trials=trials_nn)

        # Map indices to activation functions
        activation_fn_mapping = {
            0: nn.ReLU,
            1: nn.Sigmoid,
            2: nn.LeakyReLU,
            3: nn.ELU,
            4: nn.Tanh,
            5: nn.SELU
        }
        activation_fn = activation_fn_mapping[best_params_nn['activation_fn']]

        # Evaluate the full space for hidden layers using the returned best index
        evaluated_space = space_eval(search_space_nn, best_params_nn)

        # Extract neurons list
        neurons_list = evaluated_space['hidden_layers']  # This will now be the fully resolved list of neurons

        # Build the final parameter dictionary
        best_params = {
            'Hidden layer': neurons_list,
            'Dropout_rate': best_params_nn['dropout_rate'],
            'Activation function': activation_fn,
            'Batch_norm': best_params_nn['use_batch_norm'],
            'lr': best_params_nn['learning_rate']
        }

        print("Best hyperparameters for Deep Ensemble Neural Network:", best_params)
        return best_params

    def predict(self, X_test):
        """
        Predict using the ensemble by calculating mean and standard deviation
        across predictions from all networks.

        Parameters:
            X_test (array-like): Test dataset.

        Returns:
            tuple:
                - mean_prediction (np.ndarray): Mean prediction across all networks.
                - std_prediction (np.ndarray): Standard deviation of predictions across all networks.
        """
        if self.con_probmodel == True:
            mean_allnet = np.zeros((self.num_networks, len(X_test[0,:,:])))
            std_allnet = np.zeros((self.num_networks, len(X_test[0,:,:])))
            mean_pred = np.zeros((1,len(X_test[0,:,:])))
            total_std = np.zeros((1,len(X_test[0,:,:])))
            aleatoric_std = np.zeros((1,len(X_test[0,:,:])))

            for i, net in enumerate(self.networks):
                pred = net.predict(X_test[0,:,:])
                mean = pred[0]
                std = pred[1]
                mean_allnet[i,:] = mean
                std_allnet[i,:] = std

            for i in range(len(X_test[0,:,:])):
                mean_pred[0,i] = np.mean(mean_allnet[:, i])
                total_std[0,i] = np.sqrt(np.mean(std_allnet[:, i]**2 + mean_allnet[:, i]**2) - mean_pred[0,i]**2)
                aleatoric_std[0,i] = np.mean(std_allnet[:, i])
            return mean_pred.ravel(), total_std.ravel()
        else:
            # Gather predictions from all networks
            predictions = [net.predict(X_test[0,:,:]).detach().numpy() for net in self.networks]
            
            # Convert predictions to a numpy array of shape (num_networks, num_samples, output_dim)
            predictions = np.array(predictions)  # Shape: (num_networks, num_samples, output_dim)
            
            # Calculate mean and standard deviation along the ensemble axis
            mean_prediction = np.mean(predictions, axis=0).ravel()         # Shape: (num_samples, output_dim)
            std_prediction = np.std(predictions, axis=0).ravel()           # Shape: (num_samples, output_dim)
            
            return mean_prediction, std_prediction


class Linear_Regression:
    """
    A class for implementing a simple Linear Regression model using the scikit-learn library.
    This model is used for training and predicting continuous target variables.

    Attributes:
        linear_regression (LinearRegression): An instance of the LinearRegression model from scikit-learn.
    """
    def __init__(self):
        self.linear_regression = LinearRegression()
    
    def train(self, X_train, Y_train):
        self.linear_regression.fit(X_train[0,:,:], Y_train[0,:])
        print('Training of Linear Regression model completed')
    
    def predict(self, X_test):
        return self.linear_regression.predict(X_test[0,:,:])


class ML_Models:
    """
    A class that encapsulates various machine learning models, both deterministic and probabilistic, for training, hyperparameter optimization, and prediction.
    
    Attributes:
    -----------
    X_train : np.array
        The training features.
    Y_train : np.array
        The target values for training.
    X_val : np.array
        The validation features.
    Y_val : np.array
        The target values for validation.
    X_test : np.array
        The test features.
    Y_test : np.array
        The target values for testing.
    seed : int
        A random seed for reproducibility.
    con_deter_models : bool
        A flag indicating if deterministic models are enabled.
    con_prob_models : bool
        A flag indicating if probabilistic models are enabled.
    best_params_det : dict
        A dictionary to store the best hyperparameters for deterministic models.
    best_params_prob : dict
        A dictionary to store the best hyperparameters for probabilistic models.

    Methods:
    --------
    fit_determinstic_models():
        Trains the deterministic models (Linear Regression, Random Forest, Support Vector Regression, KNN, and Neural Network).
    fit_probablistic_models():
        Trains the probabilistic models (Gaussian Process, Random Forest CI, Probabilistic Neural Network, and Deep Ensembles).
    hypopt_determinstic_models():
        Optimizes the hyperparameters for deterministic models.
    hypopt_probablistic_models():
        Optimizes the hyperparameters for probabilistic models.
    predict_determinstic_models():
        Makes predictions using the deterministic models.
    predict_probablistic_models():
        Makes predictions using the probabilistic models.
    """
    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, seed, best_params_det, best_params_prob, con_optpara, con_deter_models, con_prob_models):
        """
        Initializes the ML_Models object with the provided datasets, hyperparameters, and model configuration.
        
        Parameters:
        -----------
        X_train : np.array
            The training features.
        Y_train : np.array
            The target values for training.
        X_val : np.array
            The validation features.
        Y_val : np.array
            The target values for validation.
        X_test : np.array
            The test features.
        Y_test : np.array
            The target values for testing.
        seed : int
            A random seed for reproducibility.
        best_params_det : dict
            A dictionary of the best hyperparameters for deterministic models.
        best_params_prob : dict
            A dictionary of the best hyperparameters for probabilistic models.
        con_optpara : bool
            A flag for model optimization parameters.
        con_deter_models : bool
            A flag indicating if deterministic models should be initialized.
        con_prob_models : bool
            A flag indicating if probabilistic models should be initialized.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.seed = seed
        self.con_deter_models = con_deter_models
        self.con_prob_models = con_prob_models
        self.best_params_det = {'RF': None, 'SVR':None, 'KNN':None, 'NN':None}
        self.best_params_prob = {'GPR':None, 'RFCI':None, 'PNN':None, 'DeepEnsembles':None}

        if self.con_deter_models == True:
            self.lr = Linear_Regression()
            self.rf = RandomForest_Regressor(best_params=best_params_det['RF'], con_optpara=con_optpara)
            self.svr = SupportVector_Regressor(best_params=best_params_det['SVR'], con_optpara=con_optpara)
            self.knn = KNearestNeighbor_Regressor(best_params=best_params_det['KNN'], con_optpara=con_optpara)
            self.nn = ANN_Regressor(input_dim=self.X_train.shape[2], output_dim=1, seed=self.seed, best_params=best_params_det['NN'], con_optpara=con_optpara, con_probmodel=False, epochs=5000, early_stopping=True, patience=500)
        if self.con_prob_models == True:
            self.gpr = GaussianProcess_Regressor(num_features=self.X_train.shape[2], best_params=best_params_prob['GPR'], con_optpara=con_optpara)
            self.rfci = RandomForest_CI(best_params=best_params_prob['RFCI'], con_optpara=con_optpara)
            self.probabilistic_nn = ANN_Regressor(input_dim=self.X_train.shape[2], output_dim=1, seed=self.seed, best_params=best_params_prob['PNN'], con_optpara=con_optpara, con_probmodel=True, epochs=5000, early_stopping=True, patience=500)
            self.deep_ensemles = DeepEnsemble_Regressor(num_networks=10, input_dim=self.X_train.shape[2], output_dim=1, seed=self.seed, best_params=best_params_prob['DeepEnsembles'], con_optpara=con_optpara, con_probmodel=True, epochs=5000, early_stopping=True, patience=500)

    def fit_determinstic_models(self):
        """
        Trains the deterministic models: Linear Regression, Random Forest, Support Vector Regression, KNN, and Neural Network.

        If `con_deter_models` is set to `True`, it will train the specified models. Otherwise, it will print a message indicating that deterministic models are not initialized.
        """
        if self.con_deter_models == True:
            self.lr.train(self.X_train, self.Y_train)
            self.rf.train(self.X_train, self.Y_train)
            self.svr.train(self.X_train, self.Y_train)
            self.knn.train(self.X_train, self.Y_train)
            self.nn.train(self.X_train, self.Y_train, self.X_val, self.Y_val)
        else:
            print("Determinstic models not initialised")
    
    def fit_probablistic_models(self):
        """
        Trains the probabilistic models: Gaussian Process, Random Forest CI, Probabilistic Neural Network, and Deep Ensembles.

        If `con_prob_models` is set to `True`, it will train the specified models. Otherwise, it will print a message indicating that probabilistic models are not initialized.
        """
        if self.con_prob_models == True:
            self.gpr.train(self.X_train, self.Y_train)
            self.rfci.train(self.X_train, self.Y_train)
            self.probabilistic_nn.train(self.X_train, self.Y_train, self.X_val, self.Y_val)
            self.deep_ensemles.train(self.X_train, self.Y_train, self.X_val, self.Y_val)
        else:
            print("Probablistic models not initialised")

    def hypopt_determinstic_models(self):
        """
        Optimizes the hyperparameters for deterministic models using cross-validation.

        Returns:
        --------
        dict
            A dictionary containing the best hyperparameters for each deterministic model.
        
        If `con_deter_models` is set to `True`, it will perform hyperparameter optimization for each model. Otherwise, it will print a message indicating that deterministic models are not initialized.
        """
        if self.con_deter_models == True:
            best_params_rf = self.rf.optimize_hyperparameters(self.X_train, self.Y_train, self.X_val, self.Y_val)
            best_params_svr = self.svr.optimize_hyperparameters(self.X_train, self.Y_train, self.X_val, self.Y_val)
            best_params_knn = self.knn.optimize_hyperparameters(self.X_train, self.Y_train, self.X_val, self.Y_val)
            best_params_nn = self.nn.optimize_hyperparameters(self.X_train, self.Y_train, self.X_val, self.Y_val)

            self.best_params_det = {'RF':best_params_rf, 'SVR':best_params_svr, 'KNN':best_params_knn, 'NN':best_params_nn}
            
            return self.best_params_det
        else:
            print("Determinstic models not initialised")

    def hypopt_probablistic_models(self):
        """
        Optimizes the hyperparameters for probabilistic models using cross-validation.

        Returns:
        --------
        dict
            A dictionary containing the best hyperparameters for each probabilistic model.
        
        If `con_prob_models` is set to `True`, it will perform hyperparameter optimization for each model. Otherwise, it will print a message indicating that probabilistic models are not initialized.
        """
        if self.con_prob_models == True:
            best_params_gpr = self.gpr.optimize_hyperparameters(self.X_train, self.Y_train, self.X_val, self.Y_val)
            best_params_rfci = self.rfci.optimize_hyperparameters(self.X_train, self.Y_train, self.X_val, self.Y_val)
            best_params_pnn = self.probabilistic_nn.optimize_hyperparameters(self.X_train, self.Y_train, self.X_val, self.Y_val)
            # best_params_dpnn = self.deep_ensemles.optimize_hyperparameters(self.X_train, self.Y_train, self.X_val, self.Y_val)

            self.best_params_prob = {'GPR':best_params_gpr, 'RFCI':best_params_rfci, 'PNN':best_params_pnn, 'DeepEnsembles':best_params_pnn}

            return self.best_params_prob
        else:
            print("Probablistic models not initialised")

    def predict_determinstic_models(self):
        """
        Makes predictions using the deterministic models: Linear Regression, Random Forest, Support Vector Regression, KNN, and Neural Network.

        Returns:
        --------
        dict
            A dictionary of predictions for each deterministic model.

        If `con_deter_models` is set to `True`, it will generate predictions using each of the deterministic models. Otherwise, it will print a message indicating that deterministic models are not initialized.
        """
        if self.con_deter_models == True:
            predictions_lr = self.lr.predict(self.X_test)
            predictions_rf = self.rf.predict(self.X_test)
            predictions_svr = self.svr.predict(self.X_test)
            predictions_knn = self.knn.predict(self.X_test)
            predictions_nn = self.nn.predict(self.X_test)
            predictions_nn = predictions_nn.detach().numpy().flatten()

            pred_dic_det_mod = {'LR': np.array(predictions_lr), 'RF': np.array(predictions_rf), 'SVR': np.array(predictions_svr), 'KNN': np.array(predictions_knn), 'NN': np.array(predictions_nn)}

            return pred_dic_det_mod
        else:
            print("Determinstic models not initialised")
            return None

    def predict_probablistic_models(self):
        """
        Makes predictions using the probabilistic models: Gaussian Process, Random Forest CI, Probabilistic Neural Network, and Deep Ensembles.

        Returns:
        --------
        dict
            A dictionary of predictions for each probabilistic model.

        If `con_prob_models` is set to `True`, it will generate predictions using each of the probabilistic models. Otherwise, it will print a message indicating that probabilistic models are not initialized.
        """
        if self.con_prob_models == True:
            predictions_gpr = self.gpr.predict(self.X_test)
            predictions_rfci = self.rfci.predict(self.X_train, self.X_test)
            predictions_pnn = self.probabilistic_nn.predict(self.X_test)
            predictions_deepensembles = self.deep_ensemles.predict(self.X_test)
                
            pred_dic_prob_mod = {'GPR': predictions_gpr, 'RFCI': predictions_rfci, 'PNN': predictions_pnn, 'DeepEnsembles': predictions_deepensembles}

            return pred_dic_prob_mod
        else:
            print("Probablistic models not initialised")
            return None


class TransferLearningModel(nn.Module):
    """
    A model class that implements transfer learning with an optional probabilistic regression output. 

    Args:
        pretrained_model (nn.Module): A PyTorch model to use as the base model for transfer learning.
        con_probmodel (bool): A flag to indicate if the model should output probabilistic predictions (mean and variance).
        lr (float): The learning rate for the optimizer.
        epochs (int): The number of training epochs.
        early_stopping (bool): A flag to enable early stopping during training.
        patience (int): The number of epochs with no improvement in validation loss before early stopping is triggered.
    """
    def __init__(self, pretrained_model, con_probmodel, lr, epochs, early_stopping, patience):
        """
        Initializes the TransferLearningModel class. 

        Args:
            pretrained_model (nn.Module): A PyTorch model to use as the base model.
            con_probmodel (bool): A flag indicating if the model should output probabilistic predictions.
            lr (float): The learning rate for the optimizer.
            epochs (int): The number of epochs for training.
            early_stopping (bool): Flag to enable or disable early stopping.
            patience (int): Number of epochs to wait before triggering early stopping if validation loss does not improve.
        """
        super(TransferLearningModel, self).__init__()
        
        self.pretrained_model = pretrained_model
        # Freeze the base model layers
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        for param in self.pretrained_model.layers[-6:].parameters():
            param.requires_grad = True

        self.optimiser = optim.Adam(self.pretrained_model.layers[-6:].parameters(), lr)
        
        self.criterion = nn.MSELoss()
        self.con_probmodel = con_probmodel
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        
    def fit(self, X_train, Y_train, X_val, Y_val):
        """
        Trains the transfer learning model using the provided training and validation data.

        Args:
            X_train (Tensor): The input features for the training set.
            Y_train (Tensor): The target labels for the training set.
            X_val (Tensor): The input features for the validation set.
            Y_val (Tensor): The target labels for the validation set.

        Returns:
            None: The method trains the model and stores results in internal attributes.
        
        Raises:
            ValueError: If any of the input tensors have incompatible dimensions or types.
        
        Note:
            If early stopping is enabled, the model will stop training if the validation loss does not improve for `patience` epochs.
        """
        Y_train = Y_train.unsqueeze(-1)
        Y_val = Y_val.unsqueeze(-1)
        train_runningloss = []
        val_runningloss = []
        best_loss = float('inf')
        wait = 0

        for epoch in range(self.epochs):
            self.pretrained_model.train()

            if self.con_probmodel == True:
                mean, var = self.pretrained_model(X_train)
                train_loss = F.gaussian_nll_loss(mean, Y_train, var)
            else:
                predictions = self.pretrained_model(X_train)
                train_loss = self.criterion(predictions, Y_train)
    
            train_runningloss.append(train_loss.item())
            
            self.optimiser.zero_grad()
            train_loss.backward()
            self.optimiser.step()

            self.pretrained_model.eval()
            with torch.no_grad():
                if self.con_probmodel == True:
                    mean, var = self.pretrained_model(X_val)
                    val_loss = F.gaussian_nll_loss(mean, Y_val, var)
                else:
                    predictions = self.pretrained_model(X_val)
                    val_loss = self.criterion(predictions, Y_val)

                val_runningloss.append(val_loss.item())

            #print(f"Epoch{epoch+1}:  Training loss: {train_loss}   Validation loss: {val_loss}")

            # Check if early stopping is enabled
            if self.early_stopping == True:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_weights = copy.deepcopy(self.pretrained_model.state_dict())  # Save the best weights
                    wait = 0  # Reset patience counter
                else:
                    wait += 1  # Increment patience counter
                    #print(f'No improvement, patience counter: {wait}')

                # Check if patience limit has been reached
                if wait >= self.patience:
                    #print('Early stopping triggered')
                    self.pretrained_model.load_state_dict(best_weights)  # Restore model weights to the best observed state
                    break
    
    def predict(self, X_test):
        """
        Makes predictions using the trained model.

        Args:
            X_test (Tensor): The input features for the test set.

        Returns:
            predictions (tuple or Tensor): 
                - If `con_probmodel=True`, returns a tuple (mean, std), where `mean` is the predicted value and `std` is the standard deviation.
                - If `con_probmodel=False`, returns the model's output predictions.
        
        Note:
            The output is dependent on the `con_probmodel` flag. If probabilistic predictions are enabled, it returns both the mean and variance. Otherwise, only the point predictions are returned.
        """
        self.pretrained_model.eval()

        if self.con_probmodel == True:
            mean, var = self.pretrained_model(X_test)
            mean = mean.detach().numpy().ravel()
            std = np.sqrt(var.detach().numpy().ravel())
            predictions = (mean, std)
        else:
            predictions = self.pretrained_model(X_test)

        return predictions


    
