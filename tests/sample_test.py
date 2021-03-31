import numpy as np
import logging
import torch

from botorch import settings
settings.suppress_botorch_warnings(True)

import os, sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from accelerator_control import controller, observations, interface, parameter
from accelerator_control.algorithms import mobo, sample

import matplotlib.pyplot as plt

    
#main()
logging.basicConfig(level = logging.DEBUG)
sobo_obs = observations.TestSOBO()

c = controller.Controller('test2.json', interface = interface.TestInterface())

opt_params = c.get_named_parameters(['X1', 'X2'])


#MOBO test
mobo_obs = observations.TestMOBO()

samples = np.random.rand(10,2) * 2.0
sampler = sample.Sample(opt_params, mobo_obs.children, c, samples, normalized = False)
sampler.run(10,1)
print(c.data)


