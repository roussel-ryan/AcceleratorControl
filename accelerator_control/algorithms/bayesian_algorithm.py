import numpy as np
import torch
import logging

from . import algorithm
from .. import transformer

import botorch
from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean

from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.optim import optimize_acqf
from botorch import settings

class BayesianAlgorithm(algorithm.Algorithm):
    '''
    Wrapper to algorithm class that creates GP models for objectives and constraints

    '''

    def __init__(self, parameters, observations, controller,
                 constraints = [], **kwargs):
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

        assert isinstance(constraints, list)
        
        self.n_constraints = len(constraints)
        self.use_constraints = not self.n_constraints == 0

        super().__init__(parameters, observations + constraints, controller)

        #if defined include custom mean, covar, lk models
        self.custom_lk = kwargs.get('custom_lk', None)
        self.custom_covar = kwargs.get('custom_covar', None)

        #if we want fixed noise
        self.fixed_noise = kwargs.get('fixed_noise', None)
        if self.fixed_noise == None:
            self.use_fixed_noise = False
        else:
            self.use_fixed_noise = True

        #set to -1 for mobo
        self.f_multiplier = 1.0

        #set normalization flags based on the number of constraints
        self.f_flags = np.ones(self.n_observations)
        self.f_flags[-self.n_constraints:] = 0
        self.logger.debug(f'setting normalization flags to {self.f_flags}')
        
        
    def create_model(self):
        
        X, f = self.get_data(normalize_f = self.f_flags)
        f = self.f_multiplier * f
        
        #try to create model
        #- if Nans exist then we need to use ModelList instead of SingleTaskGP
        try:
            if self.use_fixed_noise:
                self.gp = FixedNoiseGP(X, f,
                                       torch.full_like(f, self.fixed_noise),
                                       covar_module = self.custom_covar)
            else:
                self.gp = SingleTaskGP(X, f,
                                       likelihood = self.custom_lk,
                                       covar_module = self.custom_covar)

            #fit GP hyperparameters
            mll = ExactMarginalLogLikelihood(self.gp.likelihood,
                                             self.gp)
            fit_gpytorch_model(mll)


        except botorch.exceptions.errors.InputDataError:
            #create model list
            models = []
            self.logger.debug('Nans detected, using ModelListGP instead of SingleTaskGP')
            for i in range(f.shape[1]):
                #get indexes where there are NOT NANS
                not_nan_idx = torch.nonzero(~torch.isnan(f[:,i]))            
                train_f = f[not_nan_idx,i]
                train_x = X[not_nan_idx].squeeze()

                
                
                if self.use_fixed_noise:
                    model = FixedNoiseGP(train_x, train_f,
                                         torch.full_like(train_f),
                                                         self.fixed_noise),
                                         covar_module = self.custom_covar)
                else:
                    model = SingleTaskGP(train_x, train_f,
                                         likelihood = self.custom_lk,
                                         covar_module = self.custom_covar)

                #fit GP hyperparameters
                mll = ExactMarginalLogLikelihood(model.likelihood,
                                                 model)
                fit_gpytorch_model(mll)

                #add the model to the list
                models += [model]

                
            #create list model
            self.gp = ModelListGP(*models)
                
                
        
        return self.gp

    def plot_model(self, obj_idx):
        #NOTE: ONLY WORKS FOR 2D INPUT SPACES

        X, f = self.get_data(normalize_f = self.f_flags)
        
        fig, ax = plt.subplots()

        n = 25
        x = [np.linspace(0, 1, n) for e in [0,1]]

        xx,yy = np.meshgrid(*x)
        pts = np.vstack((xx.ravel(), yy.ravel())).T
        pts = torch.from_numpy(pts).float()

        #get data where there are NOT NANS
        not_nan_idx = torch.nonzero(~torch.isnan(f[:,obj_idx]))            
        train_f = f[not_nan_idx, obj_idx]
        train_x = X[not_nan_idx].squeeze()

        
        with torch.no_grad():
            pred = self.gp.posterior(pts)
            f = pred.mean

        c = ax.pcolor(xx,yy,f[:,obj_idx].detach().reshape(n,n))
        ax.plot(*train_x.T,'+')

        fig.colorbar(c)
        

    
