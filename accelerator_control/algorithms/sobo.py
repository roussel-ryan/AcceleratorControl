import numpy as np
import torch

from . import bayesian_algorithm

from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.objective import ScalarizedObjective
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from advanced_acquisition import binary_constraint, combine_acquisition

class SingleObjectiveBayesian(bayesian_algorithm.BayesianAlgorithm):
    '''
    Conduct single objective bayesian optimization of accelerator using 
    parameters and objective objects

    '''

    def __init__(self, parameters, objective, controller, constraints = [], maximize = False, **kwargs):
        '''
        Initialize optimizer

        Arguments
        ---------
        parameters : list of parameters
             List of parameter.Parameter() objects used for optimization

        objectives : list of observations_list.Observation
             Objective objects to minimize

        controller : controller.AWAController
             Controller object used to control the accelerator
 
        constraints : list, optional
            List of binary constraint observations_list, 1.0 if constraint is satisfied,
            0.0 otherwise

        maximize : bool, default = False
             If True minimize the objective, otherwise maximize

        '''

        
        super().__init__(parameters, [objective], controller, constraints)
        self.maximize = maximize    
        self.beta = kwargs.get('beta',2.0)

        self.logger.info('Initialized single objective acquisition function with\n'\
                         f'beta: {self.beta}\n'\
                         f'n_constraints: {self.n_constraints}')

            
        
    def acquire_point(self, model):
        #finds new canidate point based on UCB acquisition function w/ or w/o constraints
        if self.use_constraints:
            flags = torch.zeros(self.n_observations).double()
            flags[0] = 1.0
            scalarized = ScalarizedObjective(flags)
            UCB = UpperConfidenceBound(model, self.beta,
                                       objective = scalarized,
                                       maximize = self.maximize)

            #define constraints
            constrs = []
            for i in range(1, self.n_constraints + 1):
                constrs += [binary_constraint.BinaryConstraint(model,i)]
                
            acq = combine_acquisition.MultiplyAcquisitionFunction(model,
                                                                  [UCB] + constrs)

        else:
            acq = UpperConfidenceBound(model, self.beta,
                                       maximize = self.maximize)




        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
        candidate, acq_value = optimize_acqf(acq, bounds = bounds,
                                             q=1, num_restarts = 20,
                                             raw_samples = 20)
        return candidate
        
                                                    



        
