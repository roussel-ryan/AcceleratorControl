import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from accelerator_control import observations


class OTR2BeamSize(observations.GroupObservation):
    def __init__(self, additional_measurements=None, n_samples=1):
        if additional_measurements is None:
            additional_measurements = []
        outputs = ['CX', 'CY', 'XRMS', 'YRMS', 'SUM', 'IMGF']

        super().__init__('OTRBeamSize', outputs, n_samples)

        self.screen_x_size = 800
        self.screen_y_size = 800
        self.required_sigmas = 2

    def get_beam_statistics(self, controller):
        base_channel_name = 'OTRS:IN20:571'
        obs_names = ['X', 'Y', 'XRMS', 'YRMS', 'SUM']
        pv_names = [base_channel_name + ':' + ele for ele in obs_names]
        self.logger.debug(f'getting data from epics {pv_names}')
        stats = controller.interface.get_parameters(pv_names)

        # if we think that the beam is clipping or not on the screen
        bad_beam = stats[0] + self.required_sigmas * stats[2] > self.screen_x_size or \
                   stats[0] - self.required_sigmas * stats[2] < 0 or \
                   stats[1] + self.required_sigmas * stats[3] > self.screen_y_size or \
                   stats[1] - self.required_sigmas * stats[3] < 0

        if bad_beam:
            stats += [0]
            stats = np.array(stats)
            stats[:-1] = np.NAN
            self.logger.warning('bad beam detected')

        else:
            stats += [1]
            stats = np.array(stats)

        return stats.reshape(1, -1)

    def __call__(self, controller, param_dict, **kwargs):

        data = []
        for _ in range(self.n_samples):
            data += [self.get_beam_statistics(controller)]

        data = np.vstack(data).T
        data_dict = dict(zip(self.output_names, data)).update(param_dict)

        return pd.DataFrame(data_dict)


class Emittance(observations.GroupObservation):
    def __init__(self, n_samples=1):
        outputs = ['EMIT', 'EMITF']
        super(Emittance, self).__init__('Emittance', outputs, n_samples)

        self.emit_lower_bound = 0.0
        self.emit_upper_bound = 5.0e-6

    def __call__(self, controller, param_dict):
        data = []
        for _ in range(self.n_samples):
            emit = controller.interface.get_emittance()

            if (emit > self.emit_upper_bound) or (emit < self.emit_lower_bound):
                self.logger.warning(f'measured emittance {emit:.2e} outside of bounds')
                data += [np.array([np.NAN, 0]).reshape(1, 2)]
            else:
                data += [np.array([emit, 1]).reshape(1, 2)]

        data = np.vstack(data).T

        data_dict = dict(zip(self.output_names, data))
        data_dict.update(param_dict)

        return pd.DataFrame(data_dict)


class OTR2BunchLength(observations.GroupObservation):
    def __init__(self, measure_z=True):
        self.measure_z = measure_z
        outputs = ['sigma_x', 'sigma_y']
        if self.measure_z:
            outputs += ['sigma_z']

        super().__init__('OTRProfiles', outputs)

    def __call__(self, controller, **kwargs):
        """
        do bunch length measurement, use TCAV off measurement to get xrms,yrms
        - read 'OTRS:IN20:571:XRMS'
        - set TCAV0 to ON (getting PV for this)
        - read 'OTRS:IN20:571:XRMS' again
        - set TCAV0 to OFF
        """
        otr_base_pv = 'OTRS:IN20:571:'
        time.sleep(3)
        xrms = controller.interface.get_pvs([otr_base_pv + 'XRMS'])[0]
        yrms = controller.interface.get_pvs([otr_base_pv + 'YRMS'])[0]
        print('xrms:', xrms)
        print('yrms:', yrms)
        sigma_z = 0

        if self.measure_z:
            controller.interface.set_tcav(1)
            time.sleep(3)
            tcav_on_xrms = controller.interface.get_pvs([otr_base_pv + 'YRMS'])[0]
            controller.interface.set_tcav(0)

            # find quad difference to get rms bunch length
            tcav_scale = 1.0  # convert rmsx size to time (units: m/s)

            sigma_z = np.copy(tcav_on_xrms)  # np.sqrt((tcav_on_xrms**2 - xrms**2) / tcav_scale**2)
            print('sigma_z:', sigma_z)
            results = np.array((xrms, yrms, sigma_z))
        else:
            results = np.array((xrms, yrms))

        if (xrms < 0) or (yrms < 0) or (sigma_z < 100):
            self.__call__(controller, )
            print('foo')

        else:
            return pd.DataFrame(data=results.reshape(1, -1),
                                columns=self.output_names)
