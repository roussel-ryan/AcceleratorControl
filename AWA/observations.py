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
import image_processing

class AWAScreen(observations.GroupObservation):
    def __init__(self, target_charge = -1, charge_deviation = 0.1,
                 image_directory = None, n_samples = 1,
                 name = 'AWAScreen', additional_outputs = []):

        '''
        AWAScreen measurement class - extends GroupObservation for AWA
        
        Arguments
        ---------
        target_charge : float, optional
            Target charge for valid observation in nC, if negative ignore.
            Default: -1 (ignore)

        charge_deviation : float, optional
            Fractional deviation from target charge on ICT1 allowed for valid 
            observation. Default: 0.1

        image_directory : str, optional
            Location to save image data. Default: None

        n_samples : int, optional
            Number of samples to take. Default: 1

        name : str, optional
            Name of observation. Default: 'AWAScreen'

        additional_outputs : list, optional
            List of strings that describe extra observations added to subclass

        '''
        
        
        self.logger = logging.getLogger(__name__)
        
        outputs = ['FWHMX',
                   'FWHMY',
                   'FWHML',
                   'CX','CY',
                   'ICT1','ICT2','ICT3','ICT4','IMGF'] + additional_outputs

        self.target_charge = target_charge
        self.charge_deviation = charge_deviation
        
        #determine if we are saving images
        self.image_directory = image_directory
        if self.image_directory == None:
            self.save_image_flag = False
        else:
            self.save_image_flag = True

            
        super().__init__(name, outputs, n_samples)

    def save_image(self, images, sdata):
        ctime = int(time.time())

        self.logger.debug(f'saving image data to {self.image_directory}')
        for i in range(self.n_samples):
            fname = f'{self.image_directory}/img_{ctime}_{i}.h5'
            with h5py.File(fname, 'w') as f:

                dset = f.create_dataset('raw', data = images[i])

                #add attrs
                #for name, item in zip(self.output_names, sdata):
                #    dset.attrs[name] = item[:,i]

    def apply_ROI(self, image, ROI_coords):
        '''
        get portion of image that is inside region of interest rectangle

        Arguments
        ---------
        image : ndarray, size (N x M)
            Image array

        ROI_coords : ndarray, size (2,2)
            Region of interest coordinates in the form
            ((x1,y1),(x2,y2)) where (x1,y1) is the lower left corner
            and (x2,y2) is the upper right corner

        '''
        return image.T[ROI_coords[0,0]:ROI_coords[1,0],
                     ROI_coords[0,1]:ROI_coords[1,1]].T

    def _get_and_check_data(self, controller):
        '''
        Get data from the interface and check its validity

        Check the following:
        - that a reasonable ROI has been specified (bigger than 2 px in each direction
        - that there is a beam in the ROI/the beam is not cut off in the ROI

        NOTE: we consider charge in the interface so that measurements can be repeated quickly

        If one of these conditions are not met then set 'IMGF' to 0 and return Nans for the 
        beamsize/location data.

        '''
        images, sdata, ROI = controller.interface.GetNewImage(self.target_charge,
                                                             self.charge_deviation,
                                                             NSamples = self.n_samples)

        
        #if image is not viable then set this flag to 0 for each image
        good_image = np.ones(len(images))
        
        
        #check ROI in both directions to make sure it is reasonable
        if np.any(ROI[1] - ROI[0] < 2):
            self.logger.warning('check your region of interest, it is very small!')
            good_image[:] = 0

        #apply ROI to images
        ROI_images = []
        for i in range(self.n_samples):
            ROI_images += [self.apply_ROI(images[i], ROI)]

            
        #check that a beam exists and is inside the ROI for each image
        for i in range(len(images)):
            if not image_processing.check_image(ROI_images[i], False):
                good_image[i] = 0

        invalid_image_idx = np.nonzero(1.0 - good_image)[0]
        self.logger.debug(f'invalid image idx: {invalid_image_idx}')
                
        #where good_image = 0 set all data elements ['FWHMX/Y/L','CX/Y']  to np.NaN
        self.logger.debug(sdata)
        self.logger.debug(f'good_image: {good_image}')
        scalar_data = np.hstack([sdata, good_image.reshape(-1,1)])
        
        for i in range(len(invalid_image_idx)):
            scalar_data[invalid_image_idx[i], np.arange(5)] = np.nan

        self.logger.debug(f'scalar data after image checking\n{scalar_data}') 
        return ROI_images, scalar_data

        
                    
    def __call__(self, controller):
        '''
        Do screen measurement using interface

        '''
        images, scalar_data = self._get_and_check_data(controller)
        
        data =  pd.DataFrame(data = scalar_data,
                             columns = self.output_names)

        self.logger.debug(f'returning dataframe:\n {data}') 
        
        if self.save_image_flag:
            self.save_image(images, scalar_data)

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
        image, sdata, ROI = controller.interface.GetNewImage(self.target_charge,
                                                             self.charge_deviation,
                                                             1)

        
        #get masked image and calculate emittance
        #masked_image = self.get_masked_image(data_pkt[0])

        #get ROI portion of the image
        image = self.get_roi(image, ROI)
        
        #calculate emittance and add to data_pkt
        emittance = emittance.calculate_emittance(image, scale,
                                                  self.slit_sep,
                                                  self.drift)
                
        scalar_data = np.hstack([*sdata,emittance])
        data =  pd.DataFrame(data = scalar_data,
                             columns = self.output_names)
        
        self.save_image(data_pkt)        

        return data
