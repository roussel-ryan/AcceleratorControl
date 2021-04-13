from abc import ABC

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


class BayesianExploration(bayesian_algorithm.BayesianAlgorithm, ABC):
    '''
    Conduct single objective bayesian exploration

    '''

    def __init__(self, parameters, observations_list, controller, constraints, **kwargs):
        '''
        Initialize optimizer

        Arguments
        ---------
        parameters : list of parameters
             List of parameter.Parameter() objects used for optimization

        observations_list : list of observations.Observation
             Observation(s)

        controller : controller.AWAController
             Controller object used to control the accelerator
 
        constraints : list of observations.Observation
             Constraint observations

        '''

        assert isinstance(observations_list, list)
        assert isinstance(constraints, list)

        self.n_observation_targets = len(observations_list)
        self.n_constraints = len(constraints)

        self.beta = kwargs.get('beta', 1e6)

        # coefficient of sigma matrix for proximal term exploration
        self.sigma = kwargs.get('sigma', 1e6)

        super().__init__(parameters, observations_list, controller, constraints, **kwargs)

    def get_acq(self, model, plotting=False):
        # finds new candidate point based on UCB acquisition function w/ or w/o constraints
        if self.use_constraints:
            self.logger.debug('using constraints')
            flags = torch.zeros(self.n_observations).double()
            n_objectives = self.n_observations - self.n_constraints
            flags[:n_objectives] = 1.0
            self.logger.debug(flags)
            scalarized = ScalarizedObjective(flags)
            ucb = UpperConfidenceBound(model, self.beta,
                                       objective=scalarized)

            # define constraints
            constrs = []
            self.logger.debug(f'n_objectives {n_objectives}')
            self.logger.debug(f'n_constraints {self.n_constraints}')

            for i in range(0, self.n_constraints):
                constrs += [binary_constraint.BinaryConstraint(model, i + n_objectives)]

            self.logger.debug(constrs[0].model.train_targets)

        else:
            self.logger.debug('NOT using constraints')
            ucb = UpperConfidenceBound(model, self.beta)
            constrs = []

        prox = proximal.ProximalAcqusitionFunction(model,
                                                   self.sigma * torch.eye(self.n_parameters))
        if plotting:
            self.plot_acq(ucb)
            self.plot_acq(constrs[0])
        acq = combine_acquisition.MultiplyAcquisitionFunction(model,
                                                              [ucb, prox] + constrs)
        return acq

    def acquire_point(self, model):
        acq = self.get_acq(model)
        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
        return self.optimize(acq, bounds)

    def optimize(self, acq, bounds):
        # sometimes this can fail due to PSDError, if it does try doing it again with
        # one less point
        try:
            candidate, acq_value = optimize_acqf(acq, bounds=bounds,
                                                 q=1, num_restarts=20,
                                                 raw_samples=20)
        except NotPSDError:
            self.logger.warning('Non PSD matrix in model, try to recreate with one less point')
            model = self.create_model(True)
            acq = self.get_acq(model)
            candidate = self.optimize(acq, bounds)

        return candidate

    def plot_acq(self, acq=None, obj_idx=0, ax=[]):
        # NOTE: ONLY WORKS FOR 2D INPUT SPACES

        x, f = self.get_data(normalize_f=self.f_flags)

        if ax is []:
            fig, ax = plt.subplots()

        if acq is None:
            acq = self.get_acq(self.gp)

        n = 25
        x = [np.linspace(0, 1, n) for e in [0, 1]]

        xx, yy = np.meshgrid(*x)
        pts = np.vstack((xx.ravel(), yy.ravel())).T
        pts = torch.from_numpy(pts).float()

        # get data where there are NOT NANS
        # not_nan_idx = torch.nonzero(~torch.isnan(f[:,obj_idx]))
        # train_f = f[not_nan_idx, obj_idx]
        # train_x = X[not_nan_idx].squeeze()
        train_x = x

        with torch.no_grad():
            f = acq.forward(pts.unsqueeze(1))

        c = ax.pcolor(xx, yy, f.reshape(n, n))
        ax.plot(*train_x.T, '+')
        ax.set_title(type(acq))
        ax.set_xlabel(self.parameter_names[0].name)
        ax.set_ylabel(self.parameter_names[1].name)
        ax.figure.colorbar(c)
