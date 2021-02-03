import numpy as np
import torch

import parameter
import observations
import utilities as utils

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import transforms  
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf



class SingleObjectiveBayesian:
    '''
    Conduct single objective bayesian optimization of accelerator using 
    parameters and objective objects

    '''

    def __init__(self, parameters, objective, controller, **kwargs):
        '''
        Initialize optimizer

        Arguments
        ---------
        parameters : list
             List of parameter.Parameter() objects used for optimization

        objective : observations.Observation
             Objective object to minimize

        controller : controller.AWAController
             Controller object used to control the accelerator
        
        minimize : bool
             If True minimize the objective, otherwise maximize

        '''
        
        assert isinstance(parameters[0], parameter.Parameter)
        assert isinstance(objective, observations.Observation)
        
        self.parameters = parameters
        self.objective = objective
        self.controller = controller

        self.parameter_names = [p.name for p in parameters]
        self.n_parameters = len(self.parameters)

        
        #check if data is available
        X, f = self.get_data()
        #contruct a GP model
        self.gp = SingleTaskGP(X,f)

        self.beta = kwargs.get('beta',0.1)

    def optimize(self, n_steps = 10, n_samples = 5):
        for i in range(10):
            #get candidate for observation
            candidate = self.max_acqf()

            #unnormalize
            candidate = utils.unnormalize(candidate, self.parameters)
            
            #set parameters
            self.controller.set_parameters(candidate.flatten(), self.parameter_names)

            #do observations
            self.controller.observe(self.objective, n_samples)

            #get data and retrain gp
            X, f = self.get_data()
            
            self.gp = SingleTaskGP(X,f)
            self.fit_gp()
            
        
        
    def max_acqf(self):
        #finds new canidate point based on UCB acquisition function
        UCB = UpperConfidenceBound(self.gp, self.beta)
        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
        candidate, acq_value = optimize_acqf(UCB, bounds = bounds,
                                             q=1, num_restarts = 5,
                                             raw_samples = 20)
        return candidate

        
    def fit_gp(self):
        #fits GP model
        mll = ExactMarginalLogLikelihood(self.gp.likelihood,
                                         self.gp)
        fit_gpytorch_model(mll)
        
    def get_data(self, normalize = True):
        '''
        return numpy array with observations and convert to torch
        Note: by default input parameters are normalized from 0 to 1
        objective values are standardized to zero mean and unit variance
        
        Always use normalize = True unless doing visualization!!!

        '''

        f = self.controller.observation_data[self.objective.name]
        X = self.controller.observation_data[
            [p.name for p in self.parameters]]

        X = torch.Tensor(X.to_numpy())
        f = torch.Tensor(f.to_numpy())

        if normalize:
            #standardize f
            f = transforms.standardize(f)
            X = utils.normalize(X, self.parameters)
        f = f.reshape(-1,1)
            
        return X, f
                                                    



        
