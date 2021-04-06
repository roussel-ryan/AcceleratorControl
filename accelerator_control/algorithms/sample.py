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
    Conduct explicit setpoint measurements from numpy array

    '''

    def __init__(self, parameters, observations, controller,
                 samples, normalized = False, **kwargs):
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

        normalized : bool, optional
             Specify if the list of samples is in normalized coordinates or not
             Default: False

        '''
        assert len(parameters) == samples.shape[1]
        
        super().__init__(parameters, observations, controller,
             n_steps = len(samples),
             **kwargs)

        #check to make sure the dimensionality of samples matches
        #the number of parameters
        
        #if samples are normalized then un-normalize them
        if not normalized:
            self.logger.debug(f'normalizing samples {samples}')
            for i in range(len(self.parameters)):
                samples[:,i] = self.parameters[i].transformer.forward(
                    samples[:,i].reshape(-1,1)).flatten()

            self.logger.debug(f'normed samples {samples}')
            
        self._samples = samples
        
        
        self.meas_number = 0
        
    def acquire_point(self, model):
        candidate = self._samples[self.meas_number]
        self.meas_number += 1
        
        return torch.tensor(candidate).reshape(1,-1)

    def create_model(self):
        return None

                                                    



        
