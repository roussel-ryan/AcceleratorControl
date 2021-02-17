import numpy as np
import time
import socket
import numpy as np
import pythoncom
from win32com import client
import select

from epics import caget, caput, cainfo
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


class AWAInterface:
    UseNIFG = False
    Initialized = False
    m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    m_AWACameraHost = "127.0.0.1"
    m_AWAPGCamPort = 2019
    m_AWANIFGPort = 2029
    NewMeasurement = False
    FWHMX = 10000
    FMHMY = 10000
    FMHML = 10000
    TempVal = 0.0
    Testing = False

    
    def __init__(self,UseFrameGrabber=True, Testing=False):
        self.Testing = Testing
        self.initialize_connections(UseFrameGrabber)
        #super().__init__(self)
        
    def initialize_connections(self,UseFrameGrabber):
        #image client
        
        if(self.Testing==True):
            self.Initialized=False
        else:
            if(self.Initialized==True):
                del self.ni_frame_grabber
                del self.AWAPGCamera
                self.m_CameraClient.close()
                self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            pythoncom.CoInitialize()
            self.UseNIFG=UseFrameGrabber
            if(self.UseNIFG==False):
                self.AWAPGCamera = client.Dispatch('AWAPGCamera.application')
                self.m_CameraPort=self.m_AWAPGCamPort            
            else:
                self.ni_frame_grabber = client.Dispatch('NIFGCtrl') 
                self.m_CameraPort=self.m_AWANIFGPort
            self.m_CameraClient.connect(("127.0.0.1",self.m_CameraPort))        
            self.m_CameraClient.setblocking(0)
            
            self.Initialized=True
        #setpoint client
        #self.AWABeamlineClient = socket.socket(socket.AF_INET, socket. SOCK_STREAM)
        
    def close(self):
        #self.AWABeamlineClient.close()
        if(self.Initialized==False):
            return
        if(self.UseNIFG==True):
            del self.ni_frame_grabber
        else:
            del self.AWAPGCamera
        
    def GetImage(self):
        if(self.Initialized==True):
            if(self.UseNIFG==True):
                return np.array(self.ni_frame_grabber.GetImage())
            else:
                return np.array(self.AWAPGCamera.GetImage())
        else:
            return 0
        
    def GetNewImage(self, NSamples):
        #Connect to camera broadcasting TCP port for notifications
        #If it is timed out, then just download and return whatever image available 
        #In order to avoid the complication of TCP buffer cleanup
        #we will simply close the connection and reopen it
        NShots=0
        self.img={}
        self.FWHMX=np.empty((NSamples, 1))
        self.FWHMY=np.empty((NSamples, 1))
        self.FWHML=np.empty((NSamples, 1))
        self.CentroidX=np.empty((NSamples, 1))
        self.CentroidY=np.empty((NSamples, 1))
        if(self.Testing==True):
            return [0]
        self.m_CameraClient.close();
        self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.m_CameraClient.connect(("127.0.0.1",self.m_CameraPort))        
        self.m_CameraClient.setblocking(0)

        while (NShots<NSamples): 
            ready = select.select([self.m_CameraClient], [], [], 2)
            if ready[0]:  
                a=self.m_CameraClient.recv(1024)
                #print(a)
                b="".join(chr(x) for x in a)
                c=eval(b)
                self.FWHMX[NShots] =c['FWHMX']
                self.FWHMY[NShots]=c['FWHMY']
                self.FWHML[NShots]=c['FWHML']
                self.CentroidX[NShots]=c['CX']
                self.CentroidY[NShots]=c['CY']
                self.NewMeasurement=True
                self.img[NShots]=self.GetImage()
                NShots += 1
        return self.img
        
    # def get_PG_image(self):
    #     return np.array(self.AWAPGCamera.GetImage())

    # def get_FG_image(self):
    #     return np.array(self.ni_frame_grabber.GetImage())

    def set_parameters(self, setvals, channels):
         #power supply control settings are -10.0 to 10.0
         #for full dynamic range for bipolar PS.
         #
         self.set_beamline(channels,setvals)
         
    def set_beamline(self, params, pvvals):
        assert len(params) == len(pvvals)
        
        #check if param values are safe
        for i in range(len(params)):
            params[i].check_param_value(pvvals[i])
        
        if(self.Testing==True):
            self.TempVal = pvvals
            return
        
        for i in range(len(params)):
            caput(params[i].channel, pvvals[i])
            logging.info(f'caput {params[i].name} {pvvals[i]}')
        logging.info('set_beamline called')
        
    def test_sobo(self):
        r = np.sin(self.TempVal)
        return np.sum(r)

    def test_mobo(self):
        x = self.TempVal
        D = 2
        f1 = x[0]  # objective 1
        g = 1 + 9 * np.sum(x[1:D] / (D-1))
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h  # objective 2

        #note: botorch assumes maximization
        return -1*np.array([f1, f2])

class TestInterface(AcceleratorInterface):
    def __init__(self):
        pass
