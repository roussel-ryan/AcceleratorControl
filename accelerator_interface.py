import numpy as np
import time
import socket
import numpy as np
import pythoncom
from win32com import client
import select

from epics import caget, caput, cainfo
import logging


class AWAInterface:
    UseNIFG=False
    Initialized=False
    m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    m_AWACameraHost="127.0.0.1"
    m_AWAPGCamPort=2019
    m_AWANIFGPort=2029
    NewMeasurement=False
    FWHMX=10000
    FMHMY=10000
    FMHML=10000
    def __init__(self):
        self.initialize_connections(True)

    def initialize_connections(self,UseFrameGrabber):
        #image client
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
        self.m_CameraClient.close();
        self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.m_CameraClient.connect(("127.0.0.1",self.m_CameraPort))        
        self.m_CameraClient.setblocking(0)
        NShots=0
        self.img={}
        self.FWHMX=np.empty((NSamples, 1))
        self.FWHMY=np.empty((NSamples, 1))
        self.FWHML=np.empty((NSamples, 1))
        self.CentroidX=np.empty((NSamples, 1))
        self.CentroidY=np.empty((NSamples, 1))

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
         set_beamline(channels,setvals)
         
    def set_beamline(self, pvnames,pvvals):
        assert len(pvnames) == len(pvvals)
        for i in range(len(pvnames)):
            caput(pvnames[i], pvvals[i])
            logging.info(f'caput {pvnames[i]} {pvvals[i]}')
        logging.info(f'set_beamline called')
