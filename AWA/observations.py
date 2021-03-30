import numpy as np
import copy
import time
import h5py
import sys, os
import pandas as pd
import logging
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import emittance_calculation

from accelerator_control import observations

class AWAScreen(observations.GroupObservation):
    def __init__(self, image_directory = None, n_samples = 1,
                 name = 'AWAScreen', additional_outputs = []):

        self.logger = logging.getLogger(__name__)
        
        outputs = ['FWHMX',
                   'FWHMY',
                   'FWHML',
                   'CX','CY',
                   'ICT1','ICT2','ICT3','ICT4'] + additional_outputs

        self.n_samples = 1

        #determine if we are saving images
        self.image_directory = image_directory
        if self.image_directory == None:
            self.save_image = False
        else:
            self.save_image = True

            
        super().__init__(name, outputs)

    def save_image(self, data_pkt):
        ctime = int(time.time())

        fname = f'{self.image_directory}/img_{ctime}_{i}.h5'
        self.logger.debug(f'saving image data to {fname}')
        for i in range(self.n_samples):
            with h5py.File(fname, 'w') as f:

                dset = f.create_dataset('raw', data = data_pkt[0][i])

                #add attrs
                for name, item in zip(self.output_names, data_pkt[1:]):
                    dset.attrs[name] = item[i]


    def __call__(self, controller):
        '''
        do screen measurement

        '''
        
        data_pkt = controller.interface.GetNewImage(self.n_samples)

        scalar_data = np.hstack(data_pkt[1:])
        data =  pd.DataFrame(data = scalar_data,
                             columns = self.output_names)

        self.logger.debug(f'returning dataframe:\n {data}') 
        
        if self.save_image:
            self.save_image(data_pkt)

        return data

    
class Emittance(AWAScreen):
    def __init__(self, screen_center, screen_radius,
                 slit_sep, drift, image_directory = None):
        '''
        Vertical emittance observation and calculation using multi-slit
        diagnostic

        Arguments
        ---------
        screen_center : ndarray, shape (2,)
            Screen center in pixels

        screen_radius : float
            Screen radius in pixels, corresponds to 25.4 mm

        slit_sep : float
            Slit seperation in meters
        
        drift : float
            Longitudinal distance between slits and YAG screen

        image_directory : str, Default None
            Directory to save screen images in

        '''

        self.screen_center = screen_center
        self.screen_radius = screen_radius
        self.slit_sep = slit_sep
        self.drift = drift

        #conversion from pixels to meters
        self.m_per_px = 25.4e-3 / self.screen_radius
            
        super().__init__(name = 'Emittance',
                         image_directory = image_directory,
                         additional_outputs = ['Emittance'])
        
    def get_masked_image(self, image):
        lx, ly = image.shape
        X, Y = np.ogrid[0:lx,0:ly]
        
        mask = (X - self.screen_center[0])**2 +\
            (Y - self.screen_center[1])**2 > self.screen_radius**2

        #set values in the masking region to zero
        image[mask] = 0

        #trim image (ie remove rows and columns that are all zeros)
        for i in [0,1]:
            image = image[:, np.all(np.nonzero(image), axis = 0)].T
        
        
        return image
        
    def __call__(self, controller):
        '''
        calculate emittance from screen measurement

        '''
        #wait for any previous changes to settle
        time.sleep(3)

        data_pkt = controller.interface.GetNewImage(1)

        #get masked image and calculate emittance
        masked_image = self.get_masked_image(data_pkt[0])

        #calculate emittance and add to data_pkt
        emittance = emittance.calculate_emittance(image, scale,
                                                  self.slit_sep,
                                                  self.drift)
                
        scalar_data = np.hstack(data_pkt[1:])
        data =  pd.DataFrame(data = scalar_data,
                             columns = self.output_names)
        
        self.save_image(data_pkt)        

        return data
