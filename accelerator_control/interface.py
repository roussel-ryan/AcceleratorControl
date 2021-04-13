from abc import ABC

import numpy as np
import time

import epics
import logging


class AcceleratorInterface(ABC):
    """
    low level controller_interface class used to communicate with the accelerator

    - should impliment the following via overwriting
        - __init__() method to establish connections to control computer
        - set_beamline() method to send PV's to pyEPICS et. al.
    - due to the complex nature of custom observations_list please define a specific observation class to
      get measurements, feel free to define methods here to do so

    """

    def __init__(self):
        """establish connections here"""
        pass

    def set_parameters(self, params, pvals):
        """
        set beamline PV's here

        Arguments
        ---------
        params : list
            List of parameter objects (see parameter.py)

        pvals : ndarray
            Array of parameter set points (unnormalized)

        """
        raise NotImplementedError

    def get_parameters(self, params):
        raise NotImplementedError


class TestInterface(AcceleratorInterface, ABC):
    def __init__(self):
        super().__init__()

    def set_parameters(self, params, pvvals):
        assert len(params) == len(pvvals)
        self.val = pvvals

    def get_parameters(self, params):
        return self.val

    def test_observation(self, n_samples):
        x = self.val

        # violates
        if not x[0] < 0.5:
            f1 = np.nan
            f2 = np.nan
            f3 = 0
        else:

            d = 2
            f1 = x[0]  # objective 1
            g = 1 + 9 * np.sum(x[1:d] / (d - 1))
            h = 1 - np.sqrt(f1 / g)
            f2 = g * h  # objective 2

            f3 = float(x[0] < 0.5)

        result = np.array([np.ones(n_samples) * f1,
                           np.ones(n_samples) * f2,
                           np.ones(n_samples) * f3]).T
        return result + np.random.rand(*result.shape) * 0.001
