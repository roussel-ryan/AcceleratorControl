import numpy as np
import time
import socket
import numpy as np
import pythoncom
from win32com import client
import select

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
        return epics.caget_many([names])

    def set_TCAV(self, state):
        #sets the tcav on (1) or off (0) depending on the value of state
        PV_name = None
        if state == 1:
            epics.caput(PV_name, 1)
        else:
            epics.caput(PV_name, 0)
    
class TestInterface(AcceleratorInterface):
    def __init__(self):
        pass
