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

class BayesianExploration(algorithm.Algorithm):
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
        
        super().__init__(parameters, observations + constraints, controller)
        self.beta = kwargs.get('beta',1e6)
    
    def acquire_point(self, model):
        #finds new canidate point based on exploration acquisition function

        #define constraints
        const = binary_constraint.BinaryConstraint(model,
                                                   self.n_observation_targets)

        #weight each objective the same (since the objective values don't actually matter)
        ucb_objective = ScalarizedObjective(torch.ones(model.num_outputs).double())
        UCB = UpperConfidenceBound(model, self.beta, objective = ucb_objective)

        #combine acquisition functions
        full_acq = combine_acquisition.MultiplyAcquisitionFunction(model, (UCB, const))
        
        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
        candidate, acq_value = optimize_acqf(full_acq, bounds = bounds,
                                             q=1, num_restarts = 20,
                                             raw_samples = 20)
        return candidate

    def create_model(self):
        #normalize only the objective observations
        norm_f_flags = np.zeros(self.n_observation_targets + self.n_constraints)
        norm_f_flags[:self.n_observation_targets] = 1.0

        #get data and add to GP
        X, f = self.get_data(normalize_f = norm_f_flags)
        
        self.gp = SingleTaskGP(X, f)

        #fit GP hyperparameters
        mll = ExactMarginalLogLikelihood(self.gp.likelihood,
                                         self.gp)
        fit_gpytorch_model(mll)

        return self.gp

        
                                                    



        
