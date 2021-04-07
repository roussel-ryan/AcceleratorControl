import numpy as np
import matplotlib.pyplot as plt
import torch

from . import bayesian_algorithm

from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.errors import NotPSDError

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
        
        self.beta = kwargs.get('beta',1e6)

        #coefficient of sigma matrix for proximal term exploration
        self.sigma = kwargs.get('sigma', 1e6)
        
        super().__init__(parameters, observations, controller, constraints, **kwargs)
        

    def get_acq(self, model, plotting = False):
        #finds new canidate point based on UCB acquisition function w/ or w/o constraints
        if self.use_constraints:
            self.logger.debug('using constraints')
            flags = torch.zeros(self.n_observations).double()
            n_objectives = self.n_observations - self.n_constraints
            flags[:n_objectives] = 1.0
            self.logger.debug(flags)
            scalarized = ScalarizedObjective(flags)
            UCB = UpperConfidenceBound(model, self.beta,
                                       objective = scalarized)
            
            #define constraints
            constrs = []
            self.logger.debug(f'n_objectives {n_objectives}')
            self.logger.debug(f'n_constraintss {self.n_constraints}')
            
            for i in range(0, self.n_constraints):
                constrs += [binary_constraint.BinaryConstraint(model, i + n_objectives)]
            
            self.logger.debug(constrs[0].model.train_targets)
            
        else:
            self.logger.debug('NOT using constraints')
            UCB = UpperConfidenceBound(model, self.beta)
            constrs = []

        prox = proximal.ProximalAcqusitionFunction(model,
                                                   self.sigma * torch.eye(self.n_parameters))
        if plotting:
            self.plot_acq(UCB)
            self.plot_acq(constrs[0])
        acq = combine_acquisition.MultiplyAcquisitionFunction(model,
                                                                  [UCB, prox] + constrs)
        return acq
        
    def acquire_point(self, model):
        acq = self.get_acq(model)
        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
    
        #sometimes this can fail due to PSDError, if it does try doing it again with
        #one less point
        try:
            candidate, acq_value = optimize_acqf(acq, bounds = bounds,
                                                 q=1, num_restarts = 20,
                                                 raw_samples = 20)
        except NotPSDError:
            self.logger.warning('Non PSD matrix in model, try to recreate with one less point')
            model = self.create_model(True)
            acq = self.get_acq(model)
            candidate, acq_value = optimize_acqf(acq, bounds = bounds,
                                                 q=1, num_restarts = 20,
                                                 raw_samples = 20)
            
        return candidate
    

    def plot_acq(self, acq = None, obj_idx = 0, ax = []):
        #NOTE: ONLY WORKS FOR 2D INPUT SPACES

        X, f = self.get_data(normalize_f = self.f_flags)
        
        if ax == []:
            fig, ax = plt.subplots()

        if acq == None:
            acq = self.get_acq(self.gp)

        n = 25
        x = [np.linspace(0, 1, n) for e in [0,1]]

        xx,yy = np.meshgrid(*x)
        pts = np.vstack((xx.ravel(), yy.ravel())).T
        pts = torch.from_numpy(pts).float()

        #get data where there are NOT NANS
        #not_nan_idx = torch.nonzero(~torch.isnan(f[:,obj_idx]))            
        #train_f = f[not_nan_idx, obj_idx]
        #train_x = X[not_nan_idx].squeeze()
        train_x = X

        with torch.no_grad():
            f = acq.forward(pts.unsqueeze(1)) 

        c = ax.pcolor(xx,yy,f.reshape(n,n))
        ax.plot(*train_x.T,'+-')
        ax.set_title(type(acq))
        ax.set_xlabel(self.parameter_names[0].name)
        ax.set_ylabel(self.parameter_names[1].name)
        ax.figure.colorbar(c)

        

        
                                                    



        
