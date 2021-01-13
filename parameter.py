import numpy as np
import torch
import json

class Parameter:
    def __init__(self, name, channel, bounds):
        self.name = name
        self.channel = channel
        self.bounds = bounds

def import_parameters(param_list):
    #import parameter objects and settings (channel, bounds etc.) from list of dicts
    plist = []
    for ele in param_list:
        p = Parameter(ele['name'], ele['channel'], torch.Tensor(ele['bounds']))
        plist += [p]

    return plist
    
def write_parameters(fname):
    pass
