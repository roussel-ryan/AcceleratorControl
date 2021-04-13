import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from accelerator_control import observations


# import image_processing as ip

class OTR2Profiles(observations.GroupObservation):
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
