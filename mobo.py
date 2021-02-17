import numpy as np
import torch
import logging

from . import parameter
from . import observations
from . import transformer

from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.optim import optimize_acqf
from botorch import settings

class MultiObjectiveBayesian:
    '''
    Conduct single objective bayesian optimization of accelerator using 
    parameters and objective objects, assumes minimization for both objectives

    '''

    def __init__(self, parameters, objectives, controller, **kwargs):
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

        assert isinstance(parameters[0], parameter.Parameter)

        #for objective in objectives:
        #    assert isinstance(objective, observations.Observation)
        
        self.parameters = parameters
        self.objectives = objectives
        self.controller = controller

        #self.parameter_names = [p.name for p in parameters]
        self.parameter_names = parameters
        self.n_parameters = len(self.parameters)
        self.n_objectives = len(self.objectives)
        
        #check if data is available
        #Data won't be available until observe of ctrl has been called.
        self.X, self.f = self.get_data()

        #contruct a GP model - Matern kernel w/ nu = 2.5 and ARD, GammaPriors on lengthscales and output scales
        #note: pass a custom kernel via the covar_module argument
        #note: objectives are multiplied by -1 to do minimization
        self.gp = SingleTaskGP(self.X, self.f)

        
    def optimize(self, n_steps = 10, n_samples = 5):
        for i in range(n_steps):
            
            #get candidate for observation and unnormalize
            candidate = self.max_acqf().numpy().flatten()
            unnormed_candidate = np.zeros_like(candidate)
            for i in range(self.n_parameters):
                unnormed_candidate[i] = self.parameters[i].transformer.backward(candidate[i].reshape(1,1))

            #unnormalize candidate
            #candidate = self.controller.tx.backward(candidate.numpy().reshape(1,-1))
            
            #set parameters
            self.controller.set_parameters(self.parameters, unnormed_candidate.astype(np.float32))

            #make observations necessary (possibly multiple measurements)
            #note: some measurements (test measurements, screen measurements) can be made simultaneously
            #the majority of measurements must be made in serial
            observations = []
            for obj in self.objectives:
                if obj.is_child:
                    if not obj.parent in observations:
                        observations += [obj.parent]
                else:
                    observations += [obj]

            logging.info(f'doing observations {[obs.name for obs in observations]}')
            for obs in observations:
                self.controller.observe(obs, n_samples)

            #get data and retrain gp
            self.X, self.f = self.get_data()
            
            self.gp = SingleTaskGP(self.X, self.f)
            self.fit_gp()
            
        
        
    def max_acqf(self):
        #finds new canidate point based on EHVI acquisition function
        partitioning = NondominatedPartitioning(2, Y = self.f)
        EHVI = ExpectedHypervolumeImprovement(self.gp, -1.0 * torch.ones(2), partitioning)

        bounds = torch.stack([torch.zeros(self.n_parameters),
                              torch.ones(self.n_parameters)])
        candidate, acq_value = optimize_acqf(EHVI, bounds = bounds,
                                             num_restarts = 5, q = 1,
                                             raw_samples = 20)
        return candidate

        
    def fit_gp(self):
        #fits GP model
        mll = ExactMarginalLogLikelihood(self.gp.likelihood,
                                         self.gp)
        fit_gpytorch_model(mll)
        
    def get_data(self, normalize = True):
        '''
        return numpy array with observations and convert to torch
        Note: by default input parameters are normalized from 0 to 1
        objective values are standardized to zero mean and unit variance
        
        Always use normalize = True unless doing visualization!!!

        '''
        data = self.controller.data.fillna(-np.inf).groupby(['state_idx']).max()
        
        f = data[[obj.name for obj in self.objectives]]
        f = f.to_numpy()
        X = data[[p.name for p in self.parameters]].to_numpy()
        
        if normalize:
            #standardize f
            tf = transformer.Transformer(f)
            f = tf.forward(f)

            #normalize each input vector
            X_normed = np.zeros_like(X)
            for i in range(self.n_parameters):
                X_normed[:,i] = self.parameters[i].transformer.forward(X[:,i].reshape(-1,1)).flatten()
            
            #X = self.controller.tx.forward(X)
        
        X = torch.from_numpy(X_normed)
        f = -1 * torch.from_numpy(f)

            
        return X, f
