import numpy as np
import torch

from . import algorithm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.objective import ScalarizedObjective
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from advanced_acquisition import binary_constraint, combine_acquisition

class Sample(algorithm.Algorithm):
    '''
    Conduct explicit setpoint scans in normalized corrdinates

    '''

    def __init__(self, parameters, observations, controller, samples, **kwargs):
        '''
        Initialize algorithm

        Arguments
        ---------
        parameters : list of parameters
             List of parameter.Parameter() objects used for optimization

        observations : list of observations.Observation
             Observation(s)

        controller : controller.AWAController
             Controller object used to control the accelerator
 
        constraints : list of observations.Observation
             Constraint observations

        '''

        #check to make sure the dimensionality of samples matches the number of parameters
        assert len(parameters) == samples.shape[1]
        
        super().__init__(parameters, observations, controller)
        self.samples = samples
        self.meas_number = 0
        
    def acquire_point(self, model):
        candidate = self.samples[self.meas_number]
        self.meas_number += 1
        
        return torch.tensor(candidate)

    def create_model(self):
        return None

        
                                                    



        
