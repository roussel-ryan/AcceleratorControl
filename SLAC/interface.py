import os
import sys

import epics
import numpy as np

sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from accelerator_control.interface import AcceleratorInterface


def set_tcav(state):
    # sets the tcav on (1) or off (0) depending on the value of state
    if state == 1:
        epics.caput('TCAV:IN20:490:TC0_C_1_TCTL', 1)
        epics.caput('KLYS:LI20:51:BEAMCODE1_TCTL', 1)

    else:
        epics.caput('TCAV:IN20:490:TC0_C_1_TCTL', 0)
        epics.caput('KLYS:LI20:51:BEAMCODE1_TCTL', 0)


class SLACInterface(AcceleratorInterface):
    def __init__(self):
        super(SLACInterface, self).__init__()

    @staticmethod
    def get_pvs(names):
        return epics.caget_many(names)

    @staticmethod
    def set_beamline(params, vals):
        assert len(params) == len(vals)

        for i in range(len(params)):
            epics.caput(params[i].name, vals[i])


class TestInterface(AcceleratorInterface):
    def __init__(self):
        super(TestInterface, self).__init__()

    def set_beamline(self, params, pvvals):
        assert len(params) == len(pvvals)
        self.val = pvvals

    def test_observation(self):
        x = self.val
        d = 2
        f1 = x[0]  # objective 1
        g = 1 + 9 * np.sum(x[1:d] / (d - 1))
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h  # objective 2

        return np.array([f1, f2]) + np.random.rand(2) * 0.01
