import numpy as np
import torch
import logging

from .. import parameter
from .. import observations
from .. import transformer



class Algorithm:
    '''
    Algorithm class for use with accelerator operations
    Must implement acquire_point() function.
    To do something model assisted implement create_model() function.
    
    '''

    def __init__(self, parameters, observations, controller,
                 pre_observation_function = None):
        '''
        Initialize algorithm

        Arguments
        ---------
        parameters : list of parameters
             List of parameter.Parameter objects used for optimization

        observations : list of observations.Observation
             Observation objects needed

        controller : controller.Controller
             Controller object used to control the accelerator

        pre_observation_function : callable, optional
             Optional function called between parameter changes and 
             performing observations. Default is None

        '''

        assert isinstance(parameters[0], parameter.Parameter)

        self.logger = logging.getLogger(__name__)
        
        self.parameters = parameters
        self.observations = observations
        self.controller = controller

        self.parameter_names = parameters
        self.n_parameters = len(self.parameters)
        self.n_observations = len(self.observations)

        self.pre_observation_function = pre_observation_function
        
    def create_model(self):
        '''
        create a model to be passed to an acquisition function
        else return None

        Use self.get_data() to get data for model
        '''
        return None

    def aquire_point(self, model = None):
        '''
        Algorithm used to pick the next observation point
        '''
        
        raise NotImplementedError

    def get_data(self, normalize_x = True, normalize_f = True):
        '''
        return numpy array with observations and convert to torch
        Note: by default input parameters are normalized from 0 to 1
        objective values are standardized to zero mean and unit variance
        
        Always use normalize = True unless doing visualization!!!

        '''

        
        data = self.controller.data.groupby(['state_idx']).mean()
        
        f = data[[obj.name for obj in self.observations]]
        f = f.to_numpy()
        x = data[[p.name for p in self.parameters]].to_numpy()

        self.logger.debug(f'Raw data from controller:\nX:\n{x}\nf:\n{f}')

        #by default normalize all f (set all of normalize flags to 1)
        if np.all(normalize_f):
            f_nflags = np.ones_like(f[0])
        else:
            assert isinstance(normalize_f, np.ndarray)
            f_nflags = normalize_f

        f_normed = self._apply_f_normalization(f, f_nflags)

    
        #by default normalize all x (set all of normalize flags to 1)
        if np.all(normalize_x):
            x_nflags = np.ones_like(x[0])
        else:
            assert isinstance(normalize_x, np.ndarray)
            x_nflags = normalize_x
    
        x_normed = self._apply_x_normalization(x, x_nflags)
  
        x = torch.from_numpy(x_normed)
        f = torch.from_numpy(f_normed)

        self.logger.debug(f'Scaled data from controller:\nX:\n{x}\nf:\n{f}')
        return x, f


    
    def run(self, n_steps = 10, n_samples = 5):
        '''
        run the algorithm
        '''
        self.logger.info(f'Starting algorithm run with {n_steps} steps' +\
                          f' and {n_samples} samples per step')

        for i in range(n_steps):
            self.logger.info(f'Step {i}')

            self.logger.debug('creating model')
            model = self.create_model()

            #acquire the next point to observe - in normalized space
            self.logger.debug('acquiring next point')
            candidate = self.acquire_point(model).squeeze()
            
            #unnormalize candidate
            unn_c = np.zeros_like(candidate)
            for i in range(self.n_parameters):
                unn_c[i] = self.parameters[i].transformer.backward(
                    candidate[i].numpy().reshape(1,1))

            self.logger.debug(f'unnormed candidate is {candidate}')
                        
            #set parameters
            self.logger.debug('setting parameters')
            self.controller.set_parameters(self.parameters,
                                           unn_c.astype(
                                               np.float32))

            #execute preobservation function
            if not self.pre_observation_function == None:
                self.pre_observation_function(self.controller)
            
            #make observations necessary (possibly grouped observations)
            #the majority of measurements must be made in serial
            self.logger.info('starting observations')
            required_observations = []
            for obj in self.observations:
                if obj.is_child:
                    if not obj.parent in required_observations:
                        required_observations += [obj.parent]
                else:
                    required_observations += [obj]

            logging.debug('doing observations' +\
                          f'{[obs.name for obs in required_observations]}')

            for obs in required_observations:
                self.controller.observe(obs)

            self.logger.info('observations done')

    
    def _apply_f_normalization(self, f, normalize_flags):
        '''
        by default apply standardized normalization
        - to modify transformer type implement get_f_transformers(f)
        '''

        try:
            self.f_transformers = self.get_f_transformers(f)
            
        except AttributeError:
            self.f_transformers = [
                transformer.Transformer(ele.reshape(-1,1),
                                        'standardize') for ele in f.T]

        logging.debug(f'number of f transformers: {len(self.f_transformers)}')
        
        #normalize f according to reference point
        f_normed = f.copy()
        for i in range(self.n_observations):
            if normalize_flags[i]:
                f_normed[:,i] = self.f_transformers[i].forward(
                    f[:,i].reshape(-1,1)).flatten()


        return f_normed

    def _apply_x_normalization(self, x, normalize_flags):
        '''
        by default apply normalization based on input bounds

        '''    

        #normalize each input vector
        x_normed = x.copy()
        for i in range(self.n_parameters):
            if normalize_flags[i]:
                x_normed[:,i] = self.parameters[i].transformer.forward(
                    x[:,i].reshape(-1,1)).flatten()

        return x_normed
