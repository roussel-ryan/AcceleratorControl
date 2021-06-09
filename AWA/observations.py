import logging
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import emittance_calculation

from accelerator_control import observations, image_processing


def apply_roi(image, roi_coords):
    """
    get portion of image that is inside region of interest rectangle

    Arguments
    ---------
    image : ndarray, size (N x M)
        Image array

    roi_coords : ndarray, size (2,2)
        Region of interest coordinates in the form
        ((x1,y1),(x2,y2)) where (x1,y1) is the lower left corner
        and (x2,y2) is the upper right corner

    """
    return image.T[roi_coords[0, 0]:roi_coords[1, 0], roi_coords[0, 1]:roi_coords[1, 1]].T


class AWAScreen(observations.GroupObservation):
    def __init__(self, target_charge=-1, charge_deviation=0.1,
                 image_directory='pics', n_samples=1,
                 name='AWAScreen', additional_outputs=None):

        """
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
            List of strings that describe extra observations_list added to subclass

        """

        if additional_outputs is None:
            additional_outputs = []
        self.logger = logging.getLogger(__name__)

        outputs = ['XRMS',
                   'YRMS',
                   'FWHMX',
                   'FWHMY',
                   'FWHML',
                   'CX', 'CY',
                   'ICT1', 'ICT2', 'ICT3', 'ICT4', 'IMGF'] + additional_outputs

        self.target_charge = target_charge
        self.charge_deviation = charge_deviation

        self.save_image_flag = False
        self.image_directory = image_directory

        super().__init__(name, outputs, n_samples)

    def save_images(self, images, data_dict):
        # determine if we are saving images
        if self.image_directory is None:
            self.save_image_flag = False
        else:
            self.save_image_flag = True

        self.logger.debug(f'saving image data to {self.image_directory}')
        fname = f'{self.image_directory}/img_{self.observation_index}.h5'
        with h5py.File(fname, 'w') as f:
            for name, item in data_dict.items():
                if item is not None:
                    f['/'].attrs[name] = item

            for i in range(self.n_samples):
                f.create_dataset(f'image_{i}', data=images[i])

    def _get_and_check_data(self, controller, n_blobs_required=1):
        """
        Get data from the controller_interface and check its validity

        Check the following:
        - that a reasonable roi has been specified (bigger than 2 px in each direction
        - that there is a beam in the roi/the beam is not cut off in the roi

        NOTE: we consider charge in the controller_interface so that measurements can be repeated quickly

        If one of these conditions are not met then set 'IMGF' to 0 and return Nans for the
        beamsize/location data.

        """
        images, sdata, roi = controller.interface.get_data(self.target_charge,
                                                           self.charge_deviation,
                                                           n_samples=self.n_samples)

        # if image is not viable then set this flag to 0 for each image
        good_image = np.ones(len(images))

        # check roi in both directions to make sure it is reasonable
        if np.any(roi[1] - roi[0] < 2):
            self.logger.warning('check your region of interest, it is very small!')
            good_image[:] = 0

        # apply roi to images
        roi_images = []
        for i in range(self.n_samples):
            roi_images += [apply_roi(images[i], roi)]

        # process and identify blobs in image
        # check that a beam exists and is inside the roi for each image
        min_size = 800
        xrms = []
        yrms = []
        for i in range(len(images)):
            timg, simg, n_blobs, elips, xr, yr = image_processing.process_and_fit(roi_images[i],
                                                                          min_size)
            xrms += [xr]
            yrms += [yr]
            good_image[i] = image_processing.check_image(timg, simg,
                                                         n_blobs, n_blobs_required)

        xrms = np.array(xrms).reshape(-1,1)
        yrms = np.array(yrms).reshape(-1,1)

        invalid_image_idx = np.nonzero(1.0 - good_image)[0]
        self.logger.debug(f'invalid image idx: {invalid_image_idx}')

        # where good_image = 0 set all data elements ['FWHMX/Y/L','CX/Y']  to np.NaN
        self.logger.debug(sdata)
        self.logger.debug(f'good_image: {good_image}')
        scalar_data = np.hstack([xrms, yrms, sdata, good_image.reshape(-1, 1)])

        #for i in range(len(invalid_image_idx)):
        #    scalar_data[invalid_image_idx[i], np.arange(5)] = np.nan

        self.logger.debug(f'scalar data after image checking\n{scalar_data}')
        return roi_images, scalar_data

    def __call__(self, controller, param_dict):
        """
        Do screen measurement using controller_interface

        """
        images, scalar_data = self._get_and_check_data(controller)

        data_dict = dict(zip(self.output_names, scalar_data.T))
        data_dict.update(param_dict)

        data = pd.DataFrame(data_dict)

        self.logger.debug(f'returning dataframe:\n {data}')

        if self.save_image_flag:
            self.save_images(images, data_dict)
        self.observation_index += 1
        return data


class Emittance(AWAScreen):
    def __init__(self, slit_sep, drift, screen_width_px, **kwargs):
        """
        Vertical emittance observation and calculation using multi-slit
        diagnostic

        Arguments
        ---------
        screen_center : ndarray, shape (2,)
            Screen center in pixels

        screen_radius : float
            Screen radius in pixels, corresponds to 25.4 mm

        slit_sep : float
            Slit separation in meters

        drift : float
            Longitudinal distance between slits and YAG screen

        image_directory : str, Default None
            Directory to save screen images in

        """

        self.slit_sep = slit_sep
        self.drift = drift

        # conversion from pixels to meters
        self.m_per_px = 25.4e-3 / screen_width_px

        super().__init__(name='EMIT',
                         additional_outputs=['EMIT', 'IXXI', 'IXPXPI', 'IXXPI', 'ROT_ANG'], **kwargs)

    def __call__(self, controller, param_dict, **kwargs):
        """
        calculate emittance from screen measurement

        """
        n_blobs_required = 4
        images, scalar_data = self._get_and_check_data(controller,
                                                       n_blobs_required)

        # calculate emittance and add to data_pkt
        emittances = []
        rotation_angles = []
        for i in range(len(images)):
            if scalar_data[i, -1] == 1.0:
                # rotate images to align beamlets to axis
                img, angle, n_blobs = image_processing.rotate_beamlets(images[i])
                rotation_angles += [angle]
                self.logger.info(f'rotation_angle: {rotation_angles[i]:.2f}')

                # calculate emittance
                emittances += [
                    emittance_calculation.calculate_emittance(img,
                                                              self.m_per_px,
                                                              self.slit_sep,
                                                              self.drift)]

            else:
                emittances += [np.nan * np.ones(4)]
                rotation_angles += [np.nan]

        emittances = np.array(emittances).reshape(-1, 4)
        # cut out any measurements of the emittance over 5e-7
        # emittances = np.where(emittances > 5e-7, np.nan, emittances).reshape(-1,1)

        rotation_angles = np.array(rotation_angles).reshape(-1, 1)

        scalar_data = np.hstack([scalar_data, emittances, rotation_angles])

        data_dict = dict(zip(self.output_names, scalar_data.T))
        data_dict.update(param_dict)

        data = pd.DataFrame(data_dict)

        self.observation_index += 1
        self.save_images(images, data_dict)

        return data
