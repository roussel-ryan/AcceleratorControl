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
            self.interface = accelerator_interface.AWAInterface(True,True)
            
        
        #self.data.astype({'state_idx':'int32', 'time':'int32'},copy = False)
            
    def observe(self, obs, n_samples = 1, **kwargs):
        wait_time = kwargs.get('wait_time', self.wait_time)
        
        #do observation and merge results with last input parameter state
        results = []
        for i in range(n_samples):
            results += [pd.concat([self.new_state.reset_index(drop=True), obs(self)], axis = 1)] 

        #state = self.new_state
        #tarray = np.vstack([state.to_numpy() for i in range(n_samples)])
        #tarray = np.hstack([tarray, values])
        #temp_df = pd.DataFrame(data = tarray,
        #                       columns =  self.parameter_names + obs.output_names)
        temp_df = pd.concat(results, ignore_index = True)
        temp_df['time'] = time.time()
        

        try:
            self.data = pd.concat([self.data, temp_df],
                                  ignore_index = True)

        except AttributeError:
            self.data = temp_df
        #return values

    def get_parameters(self, names):
        return [self.parameters[name] for name in names]
    
    
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
        #data type checking
        assert x.shape[0] == len(parameters)
        for param in parameters:
            assert isinstance(param, parameter.Parameter)

            
        parameter_names = [param.name for param in parameters]

        if not self.testing:
            self.logger.info(
                f'setting parameters {parameter_names} to values {x}') 

            self.interface.set_beamline(parameters,x)
            
        else:
            self.logger.info(
                f'Test: setting parameters {parameter_names}'\
                f' to values {x}')
                
            self.interface.set_beamline(parameters, x)
                
        time.sleep(self.wait_time)


        try:
            self.new_state = self.data[self.parameter_names + ['state_idx']].tail(1).copy(deep = True)
            print(self.new_state)

        except AttributeError:
            self.new_state = pd.DataFrame(np.zeros((1, self.n_parameters + 2)),
                                 columns = self.parameter_names + ['state_idx','time'])        
        

        for p, val in zip(parameters, x):
            self.new_state[p.name] = float(val)
        self.new_state['state_idx'] = self.new_state['state_idx'] + 1
            
        #self.data = pd.concat([self.data, new_state], ignore_index = True)
            

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

    def group_data(self):
        return self.data.fillna(-np.inf).groupby(['state_idx']).max()

        
    def reset(self):
        raise NotImplementedError
        #self.state = torch.empty((1,self.n_parameters))

