import numpy as np
import time

import epics
import logging

class AcceleratorInterface:
    '''
    low level interface class used to communicate with the accelerator

    - should impliment the following via overwriting
        - __init__() method to establish connections to control computer
        - set_beamline() method to send PV's to pyEPICS et. al.
    - due to the complex nature of custom observations please define a specific observation class to
      get measurements, feel free to define methods here to do so

    '''
    def __init__(self):
        '''establish connections here'''
        pass
        
    def set_beamline(self, params, pvals):
        '''
        set beamline PV's here
        
        Arguments
        ---------
        params : list
            List of parameter objects (see parameter.py)

        pvals : ndarray
            Array of parameter set points (unnormalized) 

        '''
        raise NotImplementedError


class SLACInterface:
    def __init__(self):
        pass
    
    def set_beamline(self, params, pvvals):
        assert len(params) == len(pvvals)
        
        for i in range(len(params)):
            epics.caput(params[i].name, pvvals[i])
    

    def get_PVs(self, names):

        return epics.caget_many(names)

    def set_TCAV(self, state):
        #sets the tcav on (1) or off (0) depending on the value of state
        if state == 1:
            epics.caput('TCAV:IN20:490:TC0_C_1_TCTL',1)
            epics.caput('KLYS:LI20:51:BEAMCODE1_TCTL',1)
            

        else:
            epics.caput('TCAV:IN20:490:TC0_C_1_TCTL',0)
            epics.caput('KLYS:LI20:51:BEAMCODE1_TCTL',0)
            

    
class TestInterface(AcceleratorInterface):
    def __init__(self):
        pass

    def set_beamline(self, params, pvvals):
        assert len(params) == len(pvvals)
        self.val = pvvals

    def test_observation(self):
        x = self.val
        D = 2
        f1 = x[0]  # objective 1
        g = 1 + 9 * np.sum(x[1:D] / (D-1))
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h  # objective 2

        return np.array([f1, f2]) + np.random.rand(2)*0.01
