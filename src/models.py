import parameter_container
import sklearn.neural_network
import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline
import common
import lzma
import pickle
import pathlib
from datetime import datetime
import numpy as np

class Model:
    '''A baseclass holding estimator and it's training parameteres
    
    Attributes
    ----------
    self.name_ : str | None
        Name of the current model
    
    self.estimator_ : a class implementing fit and predict methods | None
        estimator used for training and prediction

    self.param_container_ : Param_container | None
        container for holding possible parameter values
    
    self.best_params_ : list[tuple] | None
        a list containing pairs (param.display_text_, param.value_) denoting
        the best param choice during training for tracked parameters
    
    self.num_models_ : int | None
        number of estimators used

    '''
    def __init__(self) -> None:
        '''Initialize baseclass atributes to None'''
        self.name_ = None
        self.estimator_ = None
        self.param_container_ = None
        self.best_params_ = None
        self.num_models_ = None
    
    def reset(self) -> None:
        '''Reset baseclass atributes to None'''
        self.name_ = None
        self.estimator_ = None
        self.param_container_ = None
        self.best_params_ = None
        self.num_models_ = None

    def predict(self, data):
        '''Predict using self.estimator
        
        Parameters
        ----------
        data : {array-like, sparse matrix} of shape (n_samples, n_features) |
               (n_features) for singular datapoint
            The input data.
        '''
        data_ = np.array(data)
        if len(data_.shape) == 1:
            data_ = [data_]
        return self.estimator_.predict(data_)

    def fit(self, train_data, train_target, verbose=True):
        '''Fit self.estimator to data matrix `train_data` and target(s) `train_target`.

        Parameters
        ----------
        train_data : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.
        train_target : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        verbose : bool
            whether to display best parameter choices seen during training
        
        Notes
        -----
        If the estimator has been fitted previously and not reseted, then new training
        refits the estimator with the best parameters from the previous round
        '''
        if self.best_params_ is not None:
            # reuse the best params from previous round
            self.estimator_.fit(train_data, train_target)
        else:
            # Make brand new fit
            param_grid = self.param_container_.make_param_grid(num_models=self.num_models_)
            search = sklearn.model_selection.GridSearchCV(
                estimator = self.estimator_,
                param_grid = param_grid,
                cv = 5,
                refit = True,
                n_jobs = 4,
                verbose=False
            )
            search.fit(train_data, train_target)
            self.estimator_ = search.best_estimator_
            
            self.best_params_ = []
            for tracked_param in self.param_container_.get_tracked_params():
                if tracked_param.display_mode_ == 'chosen_param':
                    self.best_params_.append((
                        tracked_param.display_text_, search.cv_results_['param_'+tracked_param.name_.format(0)][0]
                    ))
                elif tracked_param.display_mode_ == 'all_pos_vals':
                   self.best_params_.append((
                       tracked_param.display_text_, *tracked_param.values_
                    ))

            if verbose:
                # display results for various parameter combinations
                common.display_best_params(param_container=self.param_container_, search_results=search.cv_results_)

    def save(self, folder:str="pretrained_models/", filename=None) -> None:
        '''Saves the model to the location: cwd/folder/filename

        Parameters
        ----------
        folder : str
            folder where model will be saved

        filename : str | None
            if None, then model is saved to the location
            cwd/folder/self.name_{timestamp}.model where {timestamp} denotes string
            with the format _YYYY_MM_DD_hh_mm_ss

            else model is saved to the location cwd/folder/filename

        Notes
        -----
        cwd = current working directory, i.e. folder where the main script is located
        '''
        if filename is not None:
            with lzma.open(pathlib.Path(folder).joinpath(filename), "wb") as model_file:
                pickle.dump([self.name_, self.estimator_, self.param_container_, self.best_params_, self.num_models_], model_file)
        else:
            with lzma.open(pathlib.Path(folder).joinpath(datetime.now().strftime(self.name_ + '_%Y_%m_%d_%H_%M_%S.model')), "wb") as model_file:
                pickle.dump([self.name_, self.estimator_, self.param_container_, self.best_params_, self.num_models_], model_file)

    def load(self, folder:str="pretrained_models/", filename=None) -> None:
        '''Loads the model from file located at cwd/folder/filename

        Parameters
        ----------
        folder : str
            folder where model save is located

        filename : str | None
            model is loaded from the location cwd/folder/filename

        Notes
        -----
        cwd = current working directory, i.e. folder where the main script is located
        '''
        try:
            with lzma.open(pathlib.Path(folder).joinpath(filename), "rb") as model_file:
                [self.name_, self.estimator_, self.param_container_, self.best_params_, self.num_models_] = pickle.load(model_file)
        except:
            print("Invalid filename or folder path.")

class MLP_classifier_ensemble(Model):
    '''Multi Layer Perceptron classifier ensemble

    Attributes
        ----------
        self.name_ : str
            "MLP_ensemble"
        
        self.estimator_ : sklearn.ensemble.VotingClassifier
            MLP classifier ensemble, MLP classifiers are named MLP_0, MLP_1, ...

        self.param_container_ : Param_container
            container for holding possible parameter values
        
        self.best_params_ : list[tuple] | None
            a list containing pairs (param.display_text_, param.value_) denoting
            the best param choice during training for tracked parameters
        
        self.num_models_ : int
            number of estimators in the ensemble
    '''
    def __init__(self, num_models:int=5, dataset_file=None) -> None:
        '''initializes Model as Multi Layer Perceptron classifier ensemble

        Parameters
        ----------
        num_models : int
            Number of classifiers in the ensemble (more classifiers give
            better results)
        '''
        super().__init__()

        # name of the model, used for saving it
        self.name_ = "MLP_classifier_ensemble"
        self.num_models_ = num_models

        self.estimator_ = sklearn.ensemble.VotingClassifier([
            ("MLP_{}".format(i), sklearn.neural_network.MLPClassifier())
            for i in range(num_models)
        ], voting="soft")

        # NOTE: You can change parameter combinations here:
        # all possible parameters with brief explanation can be found here:
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        self.param_container_ = parameter_container.Param_container([
            # sizes of hidden layers, more layery or bigger sizes = more estimator capacity,
            # i.e. the estimator can make more fine grained distinctions. Way too much layers
            # / big sizes or little layers / small sizes lead to overfitting/underfitting,
            # meaning poor predictions for data outside of the train set distribution
            ('chosen_param', 'layers:',   'MLP_{}__hidden_layer_sizes', [(32,32,32,32)]),
            # maximum number of optimizer iterations, usually 200-600 iterations are enough for convergence
            (None,           '',          'MLP_{}__max_iter',           [600]),
            # maximum number of optimizer function calls, only used when solver='lbfgs'
            (None,           '',          'MLP_{}__max_fun',            [15000]),
            # L2 penalty (regularization term) parameter. This type of regularization selects
            # for models with smallest weights which leads to less overfitting since estimator
            # has to match the train distribution more smoothly
            ('chosen_param', 'alpha:',    'MLP_{}__alpha',              [1e-2, 1e-3, 1e-4]),
            # The solver for weight optimization.
            # ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
            # ‘sgd’ refers to stochastic gradient descent.
            # ‘adam’ refers to a stochastic gradient-based optimizer
            #        proposed by Kingma, Diederik, and Jimmy Ba
            # The solver ‘adam’ works pretty well on relatively large datasets
            # (with thousands of training samples or more) in terms of both
            # training time and validation score. For small datasets, however,
            # ‘lbfgs’ can converge faster and perform better.
            # 
            # I would recommend using either `adam` or `lbfgs` since they work
            # relativelly great and don't have extra parameters that need to be
            # set.
            ('chosen_param', 'solver:',   'MLP_{}__solver',             ['adam', 'lbfgs']),
            #     Activation function for the hidden layer.
            # ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
            #             I would not recommend using this as it makes the whole prediction linear
            # ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
            # ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
            # ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
            ('chosen_param', 'act:',      'MLP_{}__activation',         ['tanh', 'relu']),
            # The initial learning rate used. It controls the step-size in
            # updating the weights. Only used when solver=’sgd’ or ‘adam’.
            # usually values between 0.01 - 0.0001 give the best results
            ('chosen_param', 'LR:',       'MLP_{}__learning_rate_init', [0.001]),
            # Determines random number generation for weights and bias initialization,
            # train-test split if early stopping is used, and batch sampling
            # when solver=’sgd’ or ‘adam’. Pass an int for reproducible results
            # across multiple function calls.
            # The only thing that matters here is that every MLP classifier has different
            # seed otherwise they will bahave exactly the same removing the advantage of
            # using model ensemble
            ('all_pos_vals', 'r_states:', 'MLP_{}__random_state',       [[i*257+69 for i in range(num_models)]]),
            # Tolerance for the optimization, aka. number of significant digits for each parameter
            ('chosen_param', 'tol:',      'MLP_{}__tol',                [1e-4]),
            # Whether to print progress messages to stdout (output).
            (None,           '',          'verbose',                    [True])
        ])

class Random_forest_classifier(Model):
    def __init__(self, num_models=200) -> None:
        super().__init__()

        self.name_ = "Random_forest_classifier"
        self.num_models_ = num_models
        self.estimator_ = sklearn.ensemble.RandomForestClassifier()

        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        # The most imporatant parameters are max-depth, min_samples_split, min_samples_leaf
        self.param_container_ = parameter_container.Param_container([
            ('chosen_param', 'n_estimators:',      'n_estimators',      [num_models]),
            ('chosen_param', 'criterion:',         'criterion',         ['gini', 'entropy']),
            ('chosen_param', 'max_depth:',         'max_depth',         [3, 5, 10]),
            ('chosen_param', 'min_samples_split:', 'min_samples_split', [2, 5, 10, 20]),
            ('chosen_param', 'min_samples_leaf:',  'min_samples_leaf',  [1]),
            ('chosen_param', 'random_state:',      'random_state',      [69]),
            (None,           'verbose:',           'verbose',           [0])
        ])

class Gradient_boosting_classifier(Model):
    def __init__(self, num_models=200) -> None:
        super().__init__()

        self.name_ = "Gradient_boosting_classifier"
        self.num_models_ = num_models
        self.estimator_ = sklearn.ensemble.GradientBoostingClassifier()

        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
        # The most imporatant parameters are max-depth, min_samples_split, min_samples_leaf
        self.param_container_ = parameter_container.Param_container([
            ('chosen_param', 'loss:',              'loss',              ['deviance', 'exponential']),
            ('chosen_param', 'learning_rate:',     'learning_rate',     [0.1]),
            ('chosen_param', 'n_estimators:',      'n_estimators',      [num_models]),
            ('chosen_param', 'criterion:',         'criterion',         ['friedman_mse', 'squared_error']),
            ('chosen_param', 'min_samples_split:', 'min_samples_split', [2, 5, 10]),
            ('chosen_param', 'max_depth:',         'max_depth',         [3, 5, 10]),
            ('chosen_param', 'min_samples_leaf:',  'min_samples_leaf',  [1]),
            (None,           'random_state:',      'random_state',      [69]),
            (None,           'tol:',               'tol',               [1e-4]),
            (None,           'verbose:',           'verbose',           [0])
        ])