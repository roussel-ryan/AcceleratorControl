import numpy as np
import time
import socket
import numpy as np
import pythoncom
from win32com import client

import logging


class AWAInterface:
    def __init__(self):
        self.initialize_connections()

    def initialize_connections(self):
        #image client
        pythoncom.CoInitialize()
        self.AWAPGCamera = client.Dispatch('AWAPGCamera.application')
        self.ni_frame_grabber = client.Dispatch('NIFGCtrl') 

        #setpoint client
        self.AWABeamlineClient = socket.socket(socket.AF_INET, socket. SOCK_STREAM)
        
    def close(self):
        self.AWABeamlineClient.close()
        self.AWAPGCamera.Application.Quit()
        
    def get_PG_image(self):
        return np.array(self.AWAPGCamera.GetImage())

    def get_FG_image(self):
        return np.array(self.ni_frame_grabber.GetImage())

    def set_parameters(self, counts, channels):
        assert len(counts) == len(channels)

        settings = []
        for i in range(len(counts)):
            settings += [[channels[i], int(counts[i])]]

        cmd = f'GroupSet Drive {len(counts)}'
        self.AWABeamlineClient.connect(('192.168.0.9',80))

        for ii in range(len(counts)):
            cmd += f' \{ {settings[ii][0]} {settings[ii][1]} \}'
        cmd += ' \n'
        logging.info(f'sent command: {cmd}')

        self.AWABeamlineClient.send(cmd.encode())
        data = self.AWABeamlineClient.recv(512)
        logging.info(f'response: {data}')
