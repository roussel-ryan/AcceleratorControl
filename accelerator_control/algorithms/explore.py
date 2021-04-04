import numpy as np
import torch

from . import bayesian_algorithm

from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.objective import ScalarizedObjective
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from advanced_acquisition import binary_constraint, combine_acquisition, proximal


class BayesianExploration(bayesian_algorithm.BayesianAlgorithm):
    '''
    Conduct single objective bayesian exploration

    '''

    def __init__(self, parameters, observations, controller, constraints, **kwargs):
        '''
        Initialize optimizer

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

        assert isinstance(observations, list)
        assert isinstance(constraints, list)

        self.n_observation_targets = len(observations)
        self.n_constraints = len(constraints)
        
        super().__init__(parameters, observations, controller, constraints)
        self.beta = kwargs.get('beta',1e6)

        #coefficient of sigma matrix for proximal term exploration
        self.sigma = kwargs.get('sigma', 1e6)
        
    def acquire_point(self, model):
        #finds new canidate point based on UCB acquisition function w/ or w/o constraints
        if self.use_constraints:
            flags = torch.zeros(self.n_observations).double()
            n_objectives = self.n_observations - self.n_constraints
            flags[:n_objectives] = 1.0
            scalarized = ScalarizedObjective(flags)
            UCB = UpperConfidenceBound(model, self.beta,
                                       objective = scalarized)

            #define constraints
            constrs = []
            for i in range(n_objectives, self.n_constraints + 1):
                constrs += [binary_constraint.BinaryConstraint(model,i)]
                
        else:
            UCB = UpperConfidenceBound(model, self.beta)
            constrs = []

        prox = proximal.ProximalAcqusitionFunction(model,
                                                   self.sigma * torch.eye(self.n_parameters))
        
        acq = combine_acquisition.MultiplyAcquisitionFunction(model,
                                                                  [UCB, prox] + constrs)
        



        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
        candidate, acq_value = optimize_acqf(acq, bounds = bounds,
                                             q=1, num_restarts = 20,
                                             raw_samples = 20)
        return candidate
    


        
                                                    



        
