import transformer
import torch

class OutOfBoundsError(Exception):
    pass

def check_bounds(x, parameter):
    #check if x is inside parameter bounds
    print(x)
    print(parameter.bounds)
    if (float(x) < float(parameter.bounds[0])) or (float(x) > float(parameter.bounds[1])):
        raise OutOfBoundsError

def normalize(x, parameters):
    assert x.shape[1] == len(parameters)

    #create bounds tensor
    bounds = torch.cat([p.bounds for p in parameters])
    return b_utils.transforms.normalize(x, bounds)    

def unnormalize(x, parameters):
    assert x.shape[1] == len(parameters)
    
    #create bounds tensor
    bounds = torch.cat([p.bounds for p in parameters])
    return transforms.unnormalize(x, bounds)    
