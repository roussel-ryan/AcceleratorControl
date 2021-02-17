import numpy as np
import torch
import logging

import parameter
import observations
import utilities as utils
import transformer

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.optim import optimize_acqf
from botorch import settings



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
        settings.suppress_botorch_warnings(True)

        assert isinstance(parameters[0], parameter.Parameter)

        #for objective in objectives:
        #    assert isinstance(objective, observations.Observation)
        
        self.parameters = parameters
        self.objectives = objectives
        self.controller = controller

        #self.parameter_names = [p.name for p in parameters]
        self.parameter_names = parameters
        self.n_parameters = len(self.parameters)
        self.n_objectives = len(self.objectives)
        
        #check if data is available
        #Data won't be available until observe of ctrl has been called.
        self.X, self.f = self.get_data()

        #contruct a GP model
        self.gp = SingleTaskGP(self.X,self.f)

        
    def optimize(self, n_steps = 10, n_samples = 5):
        for i in range(n_steps):
            #get candidate for observation
            candidate = self.max_acqf()

            #unnormalize candidate
            candidate = self.controller.tx.backward(candidate.numpy().reshape(1,-1))
            
            #set parameters
            self.controller.set_parameters(self.parameters, candidate.flatten())

            #make observations necessary (possibly multiple measurements)
            #note: some measurements (test measurements, screen measurements) can be made simultaneously
            #the majority of measurements must be made in serial
            observations = []
            for obj in self.objectives:
                if obj.is_child:
                    if not obj.parent in observations:
                        observations += [obj.parent]
                else:
                    observations += [obj]

            logging.info(f'doing observations {[obs.name for obs in observations]}')
            for obs in observations:
                self.controller.observe(obs, n_samples)

            #get data and retrain gp
            self.X, self.f = self.get_data()
            
            self.gp = SingleTaskGP(self.X, self.f)
            self.fit_gp()
            
        
        
    def max_acqf(self):
        #finds new canidate point based on EHVI acquisition function
        partitioning = NondominatedPartitioning(2, Y = self.f)
        EHVI = ExpectedHypervolumeImprovement(self.gp, torch.tensor((0,0)),partitioning)

        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
        candidate, acq_value = optimize_acqf(EHVI, bounds = bounds,
                                             num_restarts = 5, q = 1,
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
        data = self.controller.data.fillna(-np.inf).groupby(['state_idx']).max()


        
        f = data[
            [obj.name for obj in self.objectives]].to_numpy().reshape(-1,self.n_objectives)
        X = data[[p.name for p in self.parameters]].to_numpy()
        
        if normalize:
            #standardize f
            tf = transformer.Transformer(f)
            f = tf.forward(f)
            X = self.controller.tx.forward(X)
        
        X = torch.from_numpy(X)
        f = torch.from_numpy(f)

            
        return X, f
