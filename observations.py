import numpy as np
import torch
import image_processing as ip


class Observation:
    '''
    Observation function class that keeps track of function name and callable
    
    Arguments
    ---------

    name : string
    
    func : callable
         Observation function f(accelerator_interface.AWAInterface) that returns 
              a given measurement

    '''
    def __init__(self, name):
        self.name = name

    def __call__(self, interface):
        raise NotImplementedError

class Test(Observation):
    def __init__(self):
        super().__init__('TestObservation')

    def __call__(self,interface):
        return torch.randn(1)
        
class RMSBeamSizeX(Observation):
    def __init__(self, screen_center, screen_radius):
        self.screen_center = screen_center
        self.screen_radius = screen_radius
        
        super().__init__('RMSBeamSizeX')

    def __call__(self, interface):
        
        
        return torch.as_tensor((5.0)) + torch.rand(1)
