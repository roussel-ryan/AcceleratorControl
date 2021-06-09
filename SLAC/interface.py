import os
import sys

import epics
import numpy as np
import matlab_wrapper

sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from accelerator_control.interface import AcceleratorInterface


class SLACInterface(AcceleratorInterface):
    def __init__(self):

        local_path = os.path.dirname(os.path.abspath(__file__))

        self.ml = Matlab()
        self.ml.session.eval("addpath('{}')".format(local_path))

        super(SLACInterface, self).__init__()

    def get_parameters(self, names):
        return epics.caget_many(names)

    def set_parameters(self, params, pvals):
        assert len(params) == len(pvals)

        for i in range(len(params)):
            epics.caput(params[i].name, pvals[i])

    @staticmethod
    def set_tcav(state):
        # sets the tcav on (1) or off (0) depending on the value of state
        if state == 1:
            epics.caput('TCAV:IN20:490:TC0_C_1_TCTL', 1)
            epics.caput('KLYS:LI20:51:BEAMCODE1_TCTL', 1)

        else:
            epics.caput('TCAV:IN20:490:TC0_C_1_TCTL', 0)
            epics.caput('KLYS:LI20:51:BEAMCODE1_TCTL', 0)

    def get_emittance(self):

        emittance = -1

        self.ml.session.eval('clearvars')  # clear variables
        self.ml.session.eval(
            '[emittance_x,emittance_y,emittance_x_std,emittance_y_std,bmag_x,bmag_y,bmag_x_std,bmag_y_std] = '
            'matlab_emittance_calc()')

        emittance_x = self.ml.session.workspace.emittance_x
        emittance_y = self.ml.session.workspace.emittance_y
        emittance_x_std = self.ml.session.workspace.emittance_x_std
        emittance_y_std = self.ml.session.workspace.emittance_y_std
        bmag_x = self.ml.session.workspace.bmag_x
        bmag_y = self.ml.session.workspace.bmag_y
        bmag_x_std = self.ml.session.workspace.bmag_x_std
        bmag_y_std = self.ml.session.workspace.bmag_y_std

        print('emittance_x', emittance_x, '+-', emittance_x_std)
        print('emittance_y', emittance_y, '+-', emittance_y_std)
        print('bmag_x', bmag_x, '+-', bmag_x_std)
        print('bmag_y', bmag_y, '+-', bmag_y_std)

        emittance_geomean = np.sqrt(emittance_x * emittance_y)  # geometric mean
        bmag_geomean = np.sqrt(bmag_x * bmag_y)  # geometric mean

        print('emittance geomean ', emittance_geomean)
        print('bmag geomean ', bmag_geomean)

        emittance_bmag = bmag_geomean * emittance_geomean
        print('emittance * bmag ', emittance_bmag)

        return emittance_geomean


class Matlab(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Matlab, cls).__new__(cls, *args, **kwargs)

        return cls._instance

    def __init__(self, *args, **kwargs):
        root = kwargs.get('root', None)
        if not root:
            root = os.getenv('MATLAB_ROOT')
        print('Starting Matlab Session')
        print('root', root)
        self.session = matlab_wrapper.MatlabSession(matlab_root=root)


class TestInterface(AcceleratorInterface):

    def __init__(self):
        super(TestInterface, self).__init__()

    def set_parameters(self, params, pvals):
        assert len(params) == len(pvals)
        self.val = pvals

    def get_parameters(self, params):
        return self.val

    def test_observation(self):
        x = self.val
        d = 2
        f1 = x[0]  # objective 1
        g = 1 + 9 * np.sum(x[1:d] / (d - 1))
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h  # objective 2

        return np.array([f1, f2]) + np.random.rand(2) * 0.01
