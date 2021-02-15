import numpy as np
import torch

import parameter
import observations
import utilities as utils
import transformer

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import transforms  
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

def do_scan(self, parameter_name, lower, upper, obs):
    '''
    1D scan of a parameter with one observation
    '''

    n_steps = self.config.get('scan_steps',5)
    n_samples = self.config.get('samples', 5)
        
    self.logger.info(f'starting scan of {parameter_name}'\
                     'with {n_steps} steps and {n_samples}'\
                     'samples per step')

    X = np.linspace(lower, upper, n_steps).reshape(-1,1)
    
    for x in X:
        self.set_parameters(x, [parameter_name])
        self.observe(obs, n_samples)


class SingleObjectiveBayesian:
    '''
    Conduct single objective bayesian optimization of accelerator using 
    parameters and objective objects

    '''

    def __init__(self, parameters, objective, controller, maximize = False, **kwargs):
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
        
        minimize : bool, default = True
             If True minimize the objective, otherwise maximize

        '''
        pkey = list(parameters.keys())
        
        assert isinstance(parameters[pkey[0]], parameter.Parameter)
        assert isinstance(objective, observations.Observation)
        
        self.parameters = parameters
        self.objective = objective
        self.controller = controller

        #self.parameter_names = [p.name for p in parameters]
        self.parameter_names = list(self.parameters.keys())
        self.n_parameters = len(self.parameters)

        self.maximize = maximize
        
        #check if data is available
        #Data won't be available until observe of ctrl has been called.
        X, f = self.get_data()

        #contruct a GP model
        self.gp = SingleTaskGP(X,f)

        self.beta = kwargs.get('beta',0.1)

    def optimize(self, n_steps = 10, n_samples = 5):
        for i in range(n_steps):
            #get candidate for observation
            candidate = self.max_acqf()

            #unnormalize candidate
            candidate = self.controller.tx.backward(candidate.numpy().reshape(1,-1))
            
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
        UCB = UpperConfidenceBound(self.gp, self.beta, maximize = self.maximize)
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

        f = self.controller.data[self.objective.name].to_numpy().reshape(-1,1)
        X = self.controller.data[
            [p for p in self.parameter_names]].to_numpy()
        
        if normalize:
            #standardize f
            tf = transformer.Transformer(f)
            f = tf.forward(f)
            X = self.controller.tx.forward(X)
        
        X = torch.from_numpy(X)
        f = torch.from_numpy(f)

            
        return X, f
                                                    



        
