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
            
        #get number of parameters/objectives to store input parameter state and observation data
        #    parameter states are stored in a torch.Tensor
        #    observation data is stored in a pandas.Dataframe object
        self.n_parameters = len(self.parameters)
        self.index_key = dict(zip([p.name for p in self.parameters],range(self.n_parameters)))
        
        self.state = torch.zeros((1, self.n_parameters))

        self.observation_data = pd.DataFrame(
            dict(zip([p.name for p in self.parameters],
                     [[] for i in range(self.n_parameters)])))
                    
        
    def do_scan(self, parameter_name, obs):
        '''
        1D scan of a parameter with one observation
        '''

        n_steps = self.config.get('scan_steps',5)
        n_samples = self.config.get('samples', 5)
        
        self.logger.info(f'starting scan of {parameter_name} with {n_steps} steps and {n_samples} samples per step')

        X = torch.linspace(0,1,n_steps).reshape(-1,1)
        
        for x in X:
            x_unnormed = utils.unnormalize(x.reshape(1,-1),
                                           [self.get_parameter(parameter_name)])
            self.set_parameters(x_unnormed, [parameter_name])
            self.observe(obs, n_samples)
    
    def observe(self, obs, n_samples, **kwargs):
        wait_time = kwargs.get('wait_time', self.wait_time)
        
        values = torch.empty((n_samples, 1))
        for i in range(n_samples):
            values[i] = obs(self)
            time.sleep(wait_time)

        state = self.state[-1]
        tarray = torch.cat([state.reshape(1,-1) for i in range(n_samples)])
        tarray = np.hstack([tarray, values.reshape(n_samples,1)])
        temp_df = pd.DataFrame(data = tarray,
                               columns = [p.name for p in self.parameters] +
                               [obs.name])
            
        self.observation_data = pd.concat([self.observation_data,temp_df],
                                          ignore_index = True)
            
        return values
    
            
    def set_parameters(self, x, parameter_names):
        '''
        set parameter values based on input x
        
        Arguments
        ---------
        x : torch.Tensor (n_parameters,)
            Counts value of input parameters
        
        parameters : list
            List of Parameter objects

        '''
        assert x.shape[0] == len(parameter_names)
        self.logger.info(f'setting parameters {parameter_names} to values {x}') 

        parameters = [self.parameters[
            self.index_key[name]] for name in parameter_names]
        
        
        if not self.testing:
            self.interface.set_parameters(x,[p.channel for p in parameters])
        
        time.sleep(self.wait_time)

        #append new state to state history
        new_state = self.state[-1].clone()
        for p,val in zip(parameters, x):
            new_state[self.index_key[p.name]] = val

        self.state = torch.cat([self.state, new_state.reshape(1,-1)])
        
        
    def get_parameter(self, name):
        return self.parameters[self.index_key[name]]
        
    def get_parameters(self, param_names):
        p = []
        param_names = list(param_names) if not isinstance(param_names,list) else param_names
        for name in param_names:
            p += [self.parameters[self.index_key[name]]]
        return p

    

    def _import_config(self, fname):
        with open(fname) as f:
            self.config = json.load(f)

        self.parameters = parameter.import_parameters(self.config['parameters'])
        self.logger.info(f'Imported parameters {[p.name for p in self.parameters]}')
        
        self.wait_time = self.config.get('wait_time',2.0)

    def reset(self):
        self.state = torch.empty((1,self.n_parameters))
