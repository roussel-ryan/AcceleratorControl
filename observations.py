import numpy as np
import torch
import os
import time
import h5py
import pandas as pd
#import image_processing as ip


class Observation:
    '''
    Observation function class that keeps track of function name and callable 
    function to do the observation.
    
    If this observation can be made simultaneously with other observations using the same process 
    pass it to a GroupObservation.add_child(). By default a observation has no parent =(.
    An example of a simultaneous observation is getting the x and y beamsize from a single image.

    If is_child is true, calls to this observation will execute the group callable method instead. 
    This is done so that optimizers can call individual observations while still collecting 
    excess data during optimization. 

    For use, overwrite __call__().
    Observation callables return a pandas DataFrame object with column names = to observation name.
    
    Arguments
    ---------
    name : string

    '''
    
    def __init__(self, name):
        self.name = name
        self.is_child = False
        
    def add_parent(self, parent):
        self.is_child = True
        self.parent = parent
            
        #self.name = '.'.join((self.parent.name, self.name))
        
    def __call__(self, controller):
        '''
        do observation

        Arguments
        ---------
        controller : controller.Controller object
        nsamples : number of samples to take
 
        '''
        if self.is_child:
            return self.parent()
        else:
            raise NotImplementedError

class GroupObservation:
    ''' 
    group class for when multiple observations can be made simultaneously 
    (for example multiple aspects of the beam can be measured at once w/ an image)
    - when a child observation is called the parent observation is called instead
    - child observation name is <parent name>.<child name>

    '''
    def __init__(self, name, output_names):
        self.name = name
        self.children = []

        self.output_names = output_names

        #add children observations
        for name in self.output_names:
            obs = Observation(name)
            self.add_child(obs)
        
        
    def __call__(self, controller):
        raise NotImplementedError

    def add_child(self, child):
        child.add_parent(self)
        self.children += [child]
    
    def get_children_names(self):
        return [child.name for child in self.children]

    
        
class AWABeamSize(GroupObservation):
    def __init__(self):
        super().__init__('AWABeamSize', ['CX', 'CY', 'FWHMX', 'FWHMY'])
        
    def __call__(self, controller):
        #self.img = controller.interface.GetNewImage(nsamples)
        cx = controller.interface.CentroidX
        cy = controller.interface.CentroidY
        fwhmx = controller.interface.FWHMX 
        fwhmy = controller.interface.FWHMY 
        #self.radius=np.sqrt(self.fwhmx*self.fwhmy)
        

        return pd.DataFrame(data = np.array([cx, cy, fwhmx, fwhmy]),
                            columns = self.output_names)
                         
        
class TestSOBO(Observation):
    '''observation class used for testing SOBO algorithm'''
    def __init__(self, name = 'TestSOBO'):
        super().__init__(name)

    def __call__(self, controller):
        val = controller.interface.test_sobo().reshape(1,1)
        return pd.DataFrame(val,
                            columns = [self.name])
        
class TestMOBO(GroupObservation):
    '''observation class used for testing MOBO algorithm'''
    def __init__(self, name = 'TestMOBO'):
        self.output_names = ['1','2']
        super().__init__(name)

        #add children observations
        for name in self.output_names:
            obs = Observation(name)
            self.add_child(obs)

    def __call__(self, controller):
        vals = controller.interface.test_mobo()
        return pd.DataFrame(np.array(vals).reshape(1,-1),
                            columns = self.get_children_names())
        
     
# class RMSBeamSizeX(Observation):
#     def __init__(self, screen_center, screen_radius):
#         self.screen_center = screen_center
#         self.screen_radius = screen_radius
        
#         super().__init__('RMSBeamSizeX')

#     def __call__(self, controller):
        
        
#         return torch.as_tensor((5.0)) + torch.rand(1)
        

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
#            image = controller.interface.get_image(self.camera_type)
            image = controller.interface.GetImage()

        fname = self.folder + '/' + self.base_fname + '_' + str(int(time.time())) + '.h5'

        with h5py.File(fname,'w') as f:
            dset = f.create_dataset('image', data = image)

            cols = controller.parameter_names
            vals = controller.data.tail(1).to_numpy()[0]
            print(cols)
            print(vals)
            
            for key, val in zip(cols, vals):
                dset.attrs[key] = val

        controller.logger.info(f'saved image to file: {fname}')
        
        return [True]

