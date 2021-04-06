import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import time
import logging
import pandas as pd

import os

from . import interface
from . import observations
from . import parameter
from . import transformer

class Controller:
    '''
    Controller class that directs measurements, parameter settings and 
    observation routines. Also stores measured values in a dataframe that 
    is easily accessable.

    '''

    def __init__(self, config_fname,
                 interface = interface.TestInterface(),
                 **kwargs):
        
        self.logger = logging.getLogger(__name__)
            
        #self.interface = kwargs.get('interface', interface.TestInterface())
        self.save_path = kwargs.get('save_path', 'data/')
        self.save_fname = kwargs.get('save_fname', 'data')

        #import configuration settings from json file
        self._import_config(config_fname)

        self.start_time = int(time.time())
        self.logger.info(f'start time is {int(self.start_time)}')
        
        #create accelerator interface object
        self.interface = interface
    
                    
    def observe(self, obs, **kwargs):
        wait_time = kwargs.get('wait_time', self.wait_time)
        
        #do observation and merge results with last input parameter state
        results = []

        try:
            self.new_state
        except AttributeError:
            self.logger.error('cannot do observation as no parameters have been set')
            return None
        
        obs_data = obs(self)

        #check to make sure that obs.__call__ outputs the right number of samples
        assert len(obs_data) == obs.n_samples
        
        new_state_info = pd.concat([
            self.new_state.reset_index(drop = True) for i in range(len(obs_data))])

        temp_df = pd.concat([new_state_info.reset_index(drop = True), obs_data], axis = 1) 
        temp_df['time'] = time.time()
        

        try:
            self.data = pd.concat([self.data, temp_df], ignore_index = True)
            self.logger.debug(f'current dataset\n {self.data}')
            
        except AttributeError:
            self.data = temp_df
        
        self.save_data()
        
    def get_named_parameters(self, names):
        '''
        get parameter objects with given names list
        
        '''
        
        return [self.parameters[self.parameter_key[name]] for name in names]
    
    
    def set_parameters(self, parameters, x):
        '''
        set parameter values based on input x
        
        Arguments
        ---------
        x : np.array (n_parameters,)
            Counts value of input parameters
        
        parameters : list
            List of Parameter objects

        '''
        #data type checking and make sure we are in bounds
        assert len(x) == len(parameters)
        for i in range(len(parameters)):
            assert isinstance(parameters[i], parameter.Parameter)#,
                #f'{parameters[i]} is type {type(parameters[i])}, not type parameter'
            
            parameters[i].check_param_value(x[i])
        
            
        parameter_names = [param.name for param in parameters]

        self.logger.info(
                f'setting parameters {parameter_names} to values {x}') 

        self.interface.set_beamline(parameters, x)        
        
        #if we have previous measurements add a new state to add to self.data DataFrame
        try:
            self.new_state = self.data[self.parameter_names +\
                                       ['state_idx']].tail(1).copy(deep = True)

        except AttributeError:
            self.new_state = pd.DataFrame(np.zeros((1, self.n_parameters + 2)),
                                 columns = self.parameter_names + ['state_idx','time'])        
        

        for p, val in zip(parameters, x):
            self.new_state[p.name] = float(val)
        self.new_state['state_idx'] = self.new_state['state_idx'] + 1
            

    def _import_config(self, fname):
        with open(fname) as f:
            self.config = json.load(f)

        self.parameters = parameter.import_parameters(self.config['parameters'])
        n_params = len(self.parameters)
        self.parameter_key = {p.name : i for i,p in zip(np.arange(n_params),
                                                        self.parameters)}
        self.parameter_names = [param.name for param in self.parameters]

        self.logger.info(f'Imported parameters {self.parameter_names}')
        self.n_parameters = len(self.parameter_names)
        
        self.wait_time = self.config.get('wait_time',2.0)

        #get normalization for each parameter
        #x = np.hstack([param.bounds.reshape(2,1) for param in self.parameters])
        #self.tx = transformer.Transformer(x)

    def group_data(self):
        return self.data.fillna(-np.inf).groupby(['state_idx']).max()

        
    def reset(self):
        raise NotImplementedError
        #self.state = torch.empty((1,self.n_parameters))
        
    
    def save_data(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.data.to_pickle(self.save_path + self.save_fname + '_' + str(self.start_time) + '.pkl')

    def load_data(self, fname):
        self.data = pd.read_pickle(fname)

