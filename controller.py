import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import time
import logging
import pandas as pd

import ScreenTools

import accelerator_interface
import observations
import parameter
import utilities as utils
import transformer

class AWAController:
    '''
    Controller class that directs measurements, parameter settings and 
    observation routines. Also stores measured values in a dataframe that 
    is easily accessable.

    '''

    def __init__(self, config_fname, testing = False):
        self.logger = logging.getLogger()

            
        #import configuration settings from json file
        self._import_config(config_fname)

        #create accelerator interface object if we are not testing
        self.testing = testing
        if not self.testing:
            self.interface = accelerator_interface.AWAInterface()
        else:
            self.interface = None
            
        
        self.data = pd.DataFrame(np.zeros((1, self.n_parameters)),
                                 columns = self.parameter_names)        
        
    def do_scan(self, parameter_name, lower, upper, obs):
        '''
        1D scan of a parameter with one observation
        '''

        n_steps = self.config.get('scan_steps',5)
        n_samples = self.config.get('samples', 5)
        
        self.logger.info(f'starting scan of {parameter_name} with {n_steps} steps and {n_samples} samples per step')

        X = np.linspace(lower, upper, n_steps).reshape(-1,1)
        
        for x in X:
            self.set_parameters(x, [parameter_name])
            self.observe(obs, n_samples)
    
    def observe(self, obs, n_samples, **kwargs):
        wait_time = kwargs.get('wait_time', self.wait_time)
        
        values = np.empty((n_samples, 1))
        for i in range(n_samples):
            values[i] = obs(self)
            time.sleep(wait_time)

        state = self.data[self.parameter_names].tail(1)
        tarray = np.vstack([state.to_numpy() for i in range(n_samples)])
        tarray = np.hstack([tarray, values.reshape(n_samples,1)])
        temp_df = pd.DataFrame(data = tarray,
                               columns =  self.parameter_names + [obs.name])
            
        self.data = pd.concat([self.data,temp_df],
                              ignore_index = True)
            
        return values
    
            
    def set_parameters(self, x, parameter_names):
        '''
        set parameter values based on input x
        
        Arguments
        ---------
        x : np.array (n_parameters,)
            Counts value of input parameters
        
        parameters : list
            List of Parameter objects

        '''
        assert x.shape[0] == len(parameter_names)
        self.logger.info(f'setting parameters {parameter_names} to values {x}') 

        parameters = [self.parameters[name] for name in parameter_names]

        #check parameter settings inside machine bounds
        try:
            for x_val, p in zip(x, parameters):
                utils.check_bounds(x_val,p)
                
            if not self.testing:
                self.interface.set_parameters(x,[p.channel for p in parameters])
        
            time.sleep(self.wait_time)

            #append new state to data
            new_state = self.data[self.parameter_names].tail(1).copy(deep = True)
            for p, val in zip(parameters, x):
                new_state[p.name] = val

            self.data = pd.concat([self.data, new_state], ignore_index = True)

        except utils.OutOfBoundsError:
            logging.warning('Out of parameter bounds!')
            

    def _import_config(self, fname):
        with open(fname) as f:
            self.config = json.load(f)

        parameters_list = parameter.import_parameters(self.config['parameters'])
        self.parameters = {p.name : p for p in parameters_list}
        self.parameter_names = list(self.parameters.keys())

        self.logger.info(f'Imported parameters {self.parameter_names}')
        self.n_parameters = len(self.parameter_names)
        
        self.wait_time = self.config.get('wait_time',2.0)

        #get normalization for each parameter
        x = np.hstack([self.parameters[ele].bounds.reshape(2,1) for ele in self.parameter_names])
        self.tx = transformer.Transformer(x)
        
    def reset(self):
        raise NotImplementedError
        #self.state = torch.empty((1,self.n_parameters))
