import torch
from botorch import utils as b_utils


def normalize(x, parameters):
    assert x.shape[0] == len(parameters)

    #create bounds tensor
    bounds = torch.cat([p.bounds for p in parameters])
    return b_utils.transforms.normalize(x, bounds)    

def unnormalize(x, parameters):
    assert x.shape[0] == len(parameters)
    
    #create bounds tensor
    bounds = torch.cat([p.bounds for p in parameters])
    return b_utils.transforms.unnormalize(x, bounds)    
