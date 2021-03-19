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

    def __init__(self, parameters, observations, controller, ref, **kwargs):
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

        self.n_observations = len(observations)
        
        #reference point - assumes minimization
        assert ref.shape[0] == self.n_observations
        self.ref = ref

        super().__init__(parameters, observations, controller)

        
        #define function transformers for each objective
        self.transformers = []
        for i in range(self.n_observations):
            bound_vals = np.array((0.0,self.ref[i])).reshape(2,1)
            self.transformers += [transformer.Transformer(bound_vals)]

        
        #contruct a GP model - Matern kernel w/ nu = 2.5 and ARD, GammaPriors on lengthscales and output scales
        #note: pass a custom kernel via the covar_module argument
        #note: objectives are multiplied by -1 to do minimization
        #self.gp = SingleTaskGP(self.X, self.f)

        #set mean module variable to -1 to improve optimization and set gradient to false
        #self.gp.mean_module.constant.data = torch.tensor((-1.0,-1.0))
        #self.gp.mean_module.constant.requires_grad = False

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

        
    def create_model(self):
        #get data and add to GP, note negative sign to do minimization
        X, f = self.get_data()
        self.gp = SingleTaskGP(X, -f)

        #fit GP hyperparameters
        mll = ExactMarginalLogLikelihood(self.gp.likelihood,
                                         self.gp)
        fit_gpytorch_model(mll)

        return self.gp

    def _apply_f_normalization(self, f):
        
        #normalize f according to reference point
        f_normed = np.zeros_like(f)
        for i in range(self.n_observations):
            f_normed[:,i] = self.transformers[i].forward(
                f[:,i].reshape(-1,1)).flatten()

        return f_normed

