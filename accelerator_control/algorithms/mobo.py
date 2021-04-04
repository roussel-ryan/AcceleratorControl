import numpy as np
import torch
import logging

from . import algorithm
from .. import transformer

from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean

from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.optim import optimize_acqf
from botorch import settings

class MultiObjectiveBayesian(algorithm.Algorithm):
    '''
    Conduct single objective bayesian optimization of accelerator using 
    parameters and objective objects, assumes minimization for both objectives

    '''

    def __init__(self, parameters, observations, controller,
                 ref, constraints = [], **kwargs):
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
        
        constraints : list, optional
            List of binary constraint observations, 1.0 if constraint is satisfied, 
            0.0 otherwise

        '''
        settings.suppress_botorch_warnings(True)

        assert isinstance(constraints, list)
                
        self.n_constraints = len(constraints)
        self.use_constraints = not self.n_constraints == 0

        if self.use_constraints:
            raise NotImplementedError('sucks to suck but this isn\'t implemented yet')
        
        #reference point - assumes minimization
        assert ref.shape[0] == len(observations)
        self.ref = ref

        super().__init__(parameters, observations + constraints, controller)

        
        #define function transformers for each objective
        self.transformers = []
        for i in range(len(observations)):
            bound_vals = np.array((0.0,self.ref[i])).reshape(2,1)
            self.transformers += [transformer.Transformer(bound_vals)]

        
    def acquire_point(self, model):
        #finds new candidate point based on EHVI acquisition function
        Y = model.train_targets.transpose(0,1)
        partitioning = NondominatedPartitioning(2, Y = Y)
        EHVI = ExpectedHypervolumeImprovement(model,
                                              -1.0 * self.ref,
                                              partitioning)

        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
        candidate, acq_value = optimize_acqf(EHVI, bounds = bounds,
                                             num_restarts = 20, q = 1,
                                             raw_samples = 20)
        return candidate

        

    def get_f_transformers(self, f):
        return self.transformers
