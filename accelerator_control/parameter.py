import numpy as np
import json
import transformer

class OutOfBoundsError(Exception):
    pass

class Parameter:
    def __init__(self, name, channel, bounds):
        self.name = name
        self.channel = channel
        self.bounds = bounds

        self.transformer = transformer.Transformer(self.bounds.reshape(-1,1))
        
    def check_param_value(self, val):
        if not (self.bounds[0] <= val and val <= self.bounds[1]):
            raise OutOfBoundsError(f'Parameter value {val} outside bounds for {self.name} : {self.bounds}')
    
        
def import_parameters(param_list):
    #import parameter objects and settings (channel, bounds etc.) from list of dicts
    plist = []
    for ele in param_list:
        p = Parameter(ele['name'], ele['channel'], np.array(ele['bounds'], dtype = np.float32))
        plist += [p]

    return plist
    
def write_parameters(fname):
    pass
