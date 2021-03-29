import numpy as np
import torch

from . import algorithm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf



class SingleObjectiveBayesian(algorithm.Algorithm):
    '''
    Conduct single objective bayesian optimization of accelerator using 
    parameters and objective objects

    '''

    def __init__(self, parameters, objective, controller, maximize = False, **kwargs):
        '''
        Initialize optimizer

        Arguments
        ---------
        parameters : list of parameters
             List of parameter.Parameter() objects used for optimization

        objectives : list of observations.Observation
             Objective objects to minimize

        controller : controller.AWAController
             Controller object used to control the accelerator
 
        minimize : bool, default = True
             If True minimize the objective, otherwise maximize

        '''

        super().__init__(parameters, [objective], controller)
        self.maximize = maximize    
        self.beta = kwargs.get('beta',2.0)
    
    def acquire_point(self, model):
        #finds new canidate point based on UCB acquisition function
        UCB = UpperConfidenceBound(model, self.beta, maximize = self.maximize)
        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
        candidate, acq_value = optimize_acqf(UCB, bounds = bounds,
                                             q=1, num_restarts = 20,
                                             raw_samples = 20)
        return candidate

    def create_model(self):
        #get data and add to GP
        X, f = self.get_data()
        self.gp = SingleTaskGP(X, f)

        #fit GP hyperparameters
        mll = ExactMarginalLogLikelihood(self.gp.likelihood,
                                         self.gp)
        fit_gpytorch_model(mll)

        return self.gp

        
                                                    



        
