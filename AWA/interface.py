import numpy as np
import sys, os

sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))
import time

from accelerator_control import interface

import socket
import pythoncom
from win32com import client
import select

from epics import caget, caput, caget_many
import logging


class AWAInterface(interface.AcceleratorInterface):
    """
    controller_interface class to connect to AWA control system

    """
    UseNIFG = False
    Initialized = False
    m_AWACameraHost = "127.0.0.1"
    m_AWAPGCamPort = 2019
    m_AWANIFGPort = 2029
    NewMeasurement = False
    FWHMX = 10000
    FMHMY = 10000
    FMHML = 10000
    TempVal = 0.0
    Testing = False

    USBDIO = client.Dispatch('USBDIOCtrl.Application')

    def __init__(self, use_frame_grabber=True, testing=False):

        super().__init__()
        self.ni_frame_grabber = client.Dispatch('NIFGCtrl')
        self.AWAPGCamera = client.Dispatch('AWAPGCamera.application')
        self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.logger = logging.getLogger(__name__)
        self.logger.info('Starting controller_interface')

        self.Testing = testing

        self.initialize_connections(use_frame_grabber)
        self.logger.info('Done')

    def initialize_connections(self, use_frame_grabber):
        # image client

        if self.Testing:
            self.Initialized = False
        else:
            if self.Initialized:
                self.logger.debug('Deleteing old objects')
                del self.ni_frame_grabber
                del self.AWAPGCamera
                self.m_CameraClient.close()
                self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            else:
                pass

            pythoncom.CoInitialize()
            self.UseNIFG = use_frame_grabber
            if not self.UseNIFG:
                self.logger.debug('Connecting to AWAPGCamera application')
                self.m_CameraPort = self.m_AWAPGCamPort
            else:
                self.logger.debug('Connecting to NI Frame Grabber application')
                self.m_CameraPort = self.m_AWANIFGPort
            self.m_CameraClient.connect(("127.0.0.1", self.m_CameraPort))
            self.m_CameraClient.setblocking(0)

            self.Initialized = True
        # setpoint client
        # self.AWABeamlineClient = socket.socket(socket.AF_INET, socket. SOCK_STREAM)

    def close(self):
        if not self.Initialized:
            return

        if self.UseNIFG:
            del self.ni_frame_grabber

        else:
            del self.AWAPGCamera

    def get_raw_image(self):
        if self.Initialized:
            if self.UseNIFG:
                return np.array(self.ni_frame_grabber.GetImage())
            else:
                return np.array(self.AWAPGCamera.GetImage)
        else:
            raise RuntimeError('Trying to retrieve an image before controller_interface is initialized!')

    def get_roi(self):
        if self.Initialized:
            if self.UseNIFG:
                module = self.ni_frame_grabber
            else:
                module = self.AWAPGCamera

            x1 = module.ROIX1
            x2 = module.ROIX2
            y1 = module.ROIY1
            y2 = module.ROIY2

            return np.array(((x1, y1), (x2, y2)))
        else:
            raise RuntimeError('Trying to retrieve an image '
                               'before controller_interface is initialized!')

    def get_data(self, target_charge=-1, charge_deviation=0.1, n_samples=1):
        """
        get new image and charge data

        Arguments
        ---------
        target_charge : float, optional
            Target charge for valid observation in nC, if negative ignore.
            Default: -1 (ignore)

        charge_deviation : float, optional
            Fractional deviation from target charge on ICT1 allowed for valid
            observation. Default: 0.1

        n_samples : int
            Number of samples to take

        note - calculations of centroids and FWHM etc. are based on a region of
        interest, which might be changed by the user!

        Connect to camera broadcasting TCP port for notifications
        If it is timed out, then just download and return whatever
        image available
        In order to avoid the complication of TCP buffer cleanup
        we will simply close the connection and reopen it
        elf.
        """

        self.logger.debug(f'taking n samples {n_samples}')

        n_shots = 0
        self.img = []
        self.charge = np.empty((n_samples, 4))
        self.CentroidY = np.empty((n_samples, 1))
        self.CentroidX = np.empty((n_samples, 1))
        self.FWHML = np.empty((n_samples, 1))
        self.FWHMY = np.empty((n_samples, 1))
        self.FWHMX = np.empty((n_samples, 1))

        if not self.Testing:
            self.logger.debug('restarting camera client')
            self.m_CameraClient.close()
            self.m_CameraClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.m_CameraClient.connect(("127.0.0.1", self.m_CameraPort))
            self.m_CameraClient.setblocking(0)

        while n_shots < n_samples:
            if not self.Testing:
                ready = select.select([self.m_CameraClient], [], [], 2)

                if ready[0]:
                    # gate measurement
                    self.USBDIO.SetReadyState(2, 1)

                    # check charge on ICT1 is within bounds or charge bounds is not
                    # specified (target charge < 0)
                    ict1_charge = np.abs(caget(f'AWAICTMon:Ch1'))
                    if (np.abs(ict1_charge - target_charge) <
                        np.abs(charge_deviation * target_charge)) or \
                            (target_charge < 0):

                        a = self.m_CameraClient.recv(1024)
                        # print(a)
                        b = "".join(chr(x) for x in a)
                        try:
                            c = eval(b)

                            self.FWHMX[n_shots] = c['FWHMX']
                            self.FWHMY[n_shots] = c['FWHMY']
                            self.FWHML[n_shots] = c['FWHML']
                            self.CentroidX[n_shots] = c['CX']
                            self.CentroidY[n_shots] = c['CY']
                            self.NewMeasurement = True
                            self.img += [self.get_raw_image()]

                            # get charge
                            for i in range(1, 5):
                                self.charge[n_shots, i - 1] = caget(
                                    f'AWAICTMon:Ch{i}')

                            # get ROI
                            roi = self.get_roi()

                            self.logger.debug(roi)

                            n_shots += 1

                        except SyntaxError:
                            self.logger.warning('sleeping!')
                            time.sleep(0.1)

                    else:
                        # if we are considering charge limits then print a warning
                        if target_charge > 0:
                            self.logger.warning(f'ICT1 charge:{ict1_charge} nC'
                                                f' is outside target range')
                            time.sleep(0.1)

                    self.USBDIO.SetReadyState(2, 0)

                else:
                    self.logger.warning('camera client not ready for data')

            else:
                # generate testing data
                img, data = self.get_test_data()
                self.img += [img]
                self.FWHMX[n_shots] = data
                self.FWHMY[n_shots] = data
                self.FWHML[n_shots] = data
                self.CentroidX[n_shots] = data
                self.CentroidY[n_shots] = data

                roi = np.array(((0, 0), (700, 700)))
                self.charge[n_shots] = np.ones(4)
                n_shots += 1

        self.img = np.array(self.img)

        # collect scalar data
        sdata = np.hstack([self.FWHMX,
                           self.FWHMY,
                           self.FWHML,
                           self.CentroidX,
                           self.CentroidY,
                           self.charge])
        
        return self.img, sdata, roi

    def set_parameters(self, parameters, pvvals):
        assert len(parameters) == len(pvvals)
        self.TempVal = pvvals

        if self.Testing:
            return

        for i in range(len(parameters)):
            self.logger.debug('sending epics command')
            self.logger.debug(f'caput {parameters[i].channel} {pvvals[i]}')
            status = caput(parameters[i].channel, pvvals[i])
            self.logger.debug(f'caput return status {status}')

    def get_parameters(self, parameters):
        if self.Testing:
            return np.random.rand(len(parameters))

        channels = [ele.channel for ele in parameters]
        vals = caget_many(channels)
        return np.array(vals)

    def get_test_data(self):
        r = np.sin(self.TempVal)
        tmp = sum(r)
        data = np.genfromtxt('test_images/onenc250200.txt', names=True)

        size = 50e-3
        bins = 700
        img, xedges, yedges = np.histogram2d(data['x'], data['y'],
                                             range=np.array(((-0.0, 1.0),
                                                             (-0.0, 1.0))) * size / 2.0,
                                             bins=(bins, bins))

        img = img / np.max(img)
        img = 0.15 * np.random.rand(*img.shape) + img

        return img, tmp
