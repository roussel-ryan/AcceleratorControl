import numpy as np
import copy
import sys, os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))


from accelerator_control import interface

import socket
import pythoncom
from win32com import client
import select

from epics import caget, caput, cainfo
import logging

class AWAInterface(interface.AcceleratorInterface):
    '''
    interface class to connect to AWA control system

    '''
    UseNIFG=False
    Initialized=False
    m_AWACameraHost="127.0.0.1"
    m_AWAPGCamPort=2019
    m_AWANIFGPort=2029
    NewMeasurement=False
    FWHMX=10000
    FMHMY=10000
    FMHML=10000
    TempVal=0.0
    Testing=False

    def __init__(self,UseFrameGrabber = True,Testing = False):
        self.logger = logging.getLogger(__name__)
        self.logger.info('Starting interface')
        self.Testing = Testing

        self.initialize_connections(UseFrameGrabber)
        self.logger.info('Done')
        
    def initialize_connections(self, UseFrameGrabber):
        #image client
        
        if self.Testing:
            self.Initialized = False
        else:
            if self.Initialized:
                self.logger.debug('Deleteing old objects')
                del self.ni_frame_grabber
                del self.AWAPGCamera
                self.m_CameraClient.close()
                self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            pythoncom.CoInitialize()
            self.UseNIFG = UseFrameGrabber
            if not self.UseNIFG:
                self.logger.debug('Connecting to AWAPGCamera application')
                self.AWAPGCamera = client.Dispatch('AWAPGCamera.application')
                self.m_CameraPort = self.m_AWAPGCamPort            
            else:
                self.logger.debug('Connecting to NI Frame Grabber application')
                self.ni_frame_grabber = client.Dispatch('NIFGCtrl') 
                self.m_CameraPort = self.m_AWANIFGPort
            self.m_CameraClient.connect(("127.0.0.1",self.m_CameraPort))
            self.m_CameraClient.setblocking(0)
            
            self.Initialized = True
        #setpoint client
        #self.AWABeamlineClient = socket.socket(socket.AF_INET, socket. SOCK_STREAM)
        
    def close(self):
        if not self.Initialized:
            return
        
        if self.UseNIFG==True:
            del self.ni_frame_grabber

        else:
            del self.AWAPGCamera
        
    def GetImage(self):
        if self.Initialized:
            if self.UseNIFG==True:
                return np.array(self.ni_frame_grabber.GetImage())
            else:
                return np.array(self.AWAPGCamera.GetImage())
        else:
            self.logger.warning('Trying to retrieve an image before interface is initialized!')
            return 0

    def GetNewImage(self, NSamples = 1):
        '''
        get new image and charge data
        
        Connect to camera broadcasting TCP port for notifications
        If it is timed out, then just download and return whatever
        image available 
        In order to avoid the complication of TCP buffer cleanup
        we will simply close the connection and reopen it
        elf.
        '''
        
        

        NShots = 0
        self.img = []

        self.FWHMX = np.empty((NSamples, 1))
        self.FWHMY = np.empty((NSamples, 1))
        self.FWHML = np.empty((NSamples, 1))

        self.CentroidX = np.empty((NSamples, 1))
        self.CentroidY = np.empty((NSamples, 1))


        if not self.Testing:
            self.logger.debug('restarting camera client')
            self.m_CameraClient.close();
            self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.m_CameraClient.connect(("127.0.0.1",self.m_CameraPort))        
            self.m_CameraClient.setblocking(0)

        while NShots < NSamples: 
            if not self.Testing:
                ready = select.select([self.m_CameraClient], [], [], 2)

                if ready[0]:  
                    a = self.m_CameraClient.recv(1024)
                    #print(a)
                    b = "".join(chr(x) for x in a)
                    c = eval(b)
                    self.FWHMX[NShots] = c['FWHMX']
                    self.FWHMY[NShots] = c['FWHMY']
                    self.FWHML[NShots] = c['FWHML']
                    self.CentroidX[NShots] = c['CX']
                    self.CentroidY[NShots] = c['CY']
                    self.NewMeasurement = True
                    self.img += [self.GetImage()]
                    NShots += 1
                else:
                    self.logger.warning('camera client not ready for data')

            else:
                img, data = self.get_test_data()
                self.img += [img]
                self.FWHMX[NShots] = data
                self.FWHMY[NShots] = data
                self.FWHML[NShots] = data
                self.CentroidX[NShots] = data
                self.CentroidY[NShots] = data
                NShots += 1

        self.img = np.array(self.img)
                
        return [self.img,
                self.FWHMX,
                self.FWHMY,
                self.FWHML,
                self.CentroidX,
                self.CentroidY]
        
    def set_parameters(self, setvals, channels):
         #power supply control settings are -10.0 to 10.0
         #for full dynamic range for bipolar PS.
         #
         self.set_beamline(channels,setvals)
         
    def set_beamline(self, pvnames, pvvals):
        assert len(pvnames) == len(pvvals)
        self.TempVal=pvvals
        
        if self.Testing:
            return

        for i in range(len(pvnames)):
            caput(pvnames[i], pvvals[i])
            self.logger.debug('sending epics command')
            self.logger.debug(f'caput {pvnames[i]} {pvvals[i]}')

    def get_test_data(self):
        r = np.sin(self.TempVal)
        tmp = sum(r)
        img = np.random.rand(20, 20)
        
        return img, tmp
