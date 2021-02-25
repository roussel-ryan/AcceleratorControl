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
    
    If this observation can be made simultaneously with other observations 
    using the same process pass it to a GroupObservation.add_child(). 
    By default a observation has no parent =(.
    
    An example of a simultaneous observation is getting the x and y beamsize 
    from a single image.

    If is_child is true, calls to this observation will execute the 
    group callable method instead. 
    
    This is done so that optimizers can call individual observations while 
    still passively collecting excess data during optimization. 

    For use, overwrite __call__().
    Observation callables return a pandas DataFrame object with 
    column names = to observation name.
    
    Arguments
    ---------
    name : string

    '''
    
    def __init__(self, name):
        self.name = name
        self.is_child = False
        
    def add_parent(self, parent):
        '''adds parent to observation -> self._call__ is not necessary'''
        
        self.is_child = True
        self.parent = parent
            
        
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
        '''
        
        impliments observation procedure, 
        should return a pandas DataFrame object

        '''
        raise NotImplementedError

    def add_child(self, child):
        child.add_parent(self)
        self.children += [child]
    
    def get_children_names(self):
        return [child.name for child in self.children]


class OTR2Profiles(GroupObservation):
    def __init__(self, measure_z = True):
        self.measure_z = measure_z
        outputs = ['sigma_x','sigma_y']
        if self.measure_z:
            outputs += ['sigma_z']
        
        super().__init__('OTRProfiles', outputs)

    def __call__(self, controller):
        '''
        do bunch length measurement, use TCAV off measurement to get xrms,yrms
        - read 'OTRS:IN20:571:XRMS'
        - set TCAV0 to ON (getting PV for this)
        - read 'OTRS:IN20:571:XRMS' again
        - set TCAV0 to OFF
        '''
        otr_base_pv = 'OTRS:IN20:571:'
        time.sleep(3)
        xrms = controller.interface.get_PVs([otr_base_pv + 'XRMS'])[0]
        yrms = controller.interface.get_PVs([otr_base_pv + 'YRMS'])[0]
        print('xrms:',xrms)
        print('yrms:',yrms)
        sigma_z = 0

        if self.measure_z:
            controller.interface.set_TCAV(1)
            time.sleep(3)
            tcav_on_xrms = controller.interface.get_PVs([otr_base_pv + 'YRMS'])[0]
            controller.interface.set_TCAV(0)

            #find quad difference to get rms bunch length
            tcav_scale = 1.0 #convert rmsx size to time (units: m/s)

            sigma_z = np.copy(tcav_on_xrms)#np.sqrt((tcav_on_xrms**2 - xrms**2) / tcav_scale**2)
            print('sigma_z:',sigma_z)
            results = np.array((xrms, yrms, sigma_z))
        else:
            results = np.array((xrms,yrms))
            
            
        if (xrms < 0) or (yrms < 0) or (sigma_z < 100):
            self.__call__(controller)
            print('foo')

        
        else:
            
            return pd.DataFrame(data = results.reshape(1,-1),
                            columns = self.output_names)
        
        
class TestSOBO(Observation):
    '''observation class used for testing SOBO algorithm'''
    def __init__(self, name = 'TestSOBO'):
        super().__init__(name)

    def __call__(self, controller):
        val = controller.interface.test_observation()
        return pd.DataFrame(val[0].reshape(1,1),
                            columns = [self.name])
        
class TestMOBO(GroupObservation):
    '''observation class used for testing MOBO algorithm'''
    def __init__(self, name = 'TestMOBO'):
        output_names = ['1','2']
        super().__init__(name, output_names)

    def __call__(self, controller):
        vals = controller.interface.test_observation()
        return pd.DataFrame(vals.reshape(1,-1),
                            columns = self.output_names)
        
     

