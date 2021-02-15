import numpy as np
import torch

import parameter
import observations
import utilities as utils
import transformer

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import EHVI
from botorch.optim import optimize_acqf


class MultiObjectiveBayesian:
    '''
    Conduct single objective bayesian optimization of accelerator using 
    parameters and objective objects, assumes minimization for both objectives

    '''

    def __init__(self, parameters, objectives, controller, **kwargs):
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
        

        '''
        pkey = list(parameters.keys())
        
        assert isinstance(parameters[pkey[0]], parameter.Parameter)

        for objective in objectives:
            assert isinstance(objective, observations.Observation)
        
        self.parameters = parameters
        self.objectives = objectives
        self.controller = controller

        #self.parameter_names = [p.name for p in parameters]
        self.parameter_names = list(self.parameters.keys())
        self.n_parameters = len(self.parameters)
        self.n_objectives = len(self.objectives)
        
        #check if data is available
        #Data won't be available until observe of ctrl has been called.
        X, f = self.get_data()

        #contruct a GP model
        self.gp = SingleTaskGP(X,f)

        
    def optimize(self, n_steps = 10, n_samples = 5):
        for i in range(n_steps):
            #get candidate for observation
            candidate = self.max_acqf()

            #unnormalize candidate
            candidate = self.controller.tx.backward(candidate.numpy().reshape(1,-1))
            
            #set parameters
            self.controller.set_parameters(candidate.flatten(), self.parameter_names)

            #do observations
            self.controller.observe(self.objectives, n_samples)

            #get data and retrain gp
            X, f = self.get_data()
            
            self.gp = SingleTaskGP(X,f)
            self.fit_gp()
            
        
        
    def max_acqf(self):
        #finds new canidate point based on EHVI acquisition function
        EHVI = EHVI(self.gp)
        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
        candidate, acq_value = optimize_acqf(EHVI, bounds = bounds,
                                             num_restarts = 5,
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

        f = self.controller.data[
            [objective.name for objective in self.objectives]].to_numpy().reshape(-1,self.n_objectives)
        X = self.controller.data[
            [p for p in self.parameter_names]].to_numpy()
        
        if normalize:
            #standardize f
            tf = transformer.Transformer(f,'standardize')
            f = tf.forward(f)
            X = self.controller.tx.forward(X)
        
        X = torch.from_numpy(X)
        f = torch.from_numpy(f)

            
        return X, f
