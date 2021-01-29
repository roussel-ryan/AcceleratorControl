import numpy as np
import torch
import image_processing as ip
import os
import time

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

    def __call__(self, controller):
        '''
        do observation

        Arguments
        ---------

        interface : 
        '''
        raise NotImplementedError

class Test(Observation):
    def __init__(self):
        super().__init__('TestObservation')

    def __call__(self, controller):
        return torch.randn(1)
        
class RMSBeamSizeX(Observation):
    def __init__(self, screen_center, screen_radius):
        self.screen_center = screen_center
        self.screen_radius = screen_radius
        
        super().__init__('RMSBeamSizeX')

    def __call__(self, controller):
        
        
        return torch.as_tensor((5.0)) + torch.rand(1)

class ImageSave(Observation):
    '''
    Saves image into h5 file with settings/time metadata
    '''
    
    def __init__(self, folder, base_fname = 'image', camera_type = 'PG'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.folder = folder
        self.base_fname = base_fname
        self.camera_type = camera_type

        super().__init__('ImageSave')
        
    def __call__(self, controller):
        #get image using controller interface
        if controller.testing:
            image = np.ones((700,700))
        else:
            image = controller.interface.get_image(self.camera_type)

        fname = self.folder + '/' + self.base_fname + '_' + str(int(time.time())) + '.h5'
        controller.logger.info(f'saving image file to {fname}')

        return torch.tensor([float('nan')])

