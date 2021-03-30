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
logging.basicConfig(level = logging.INFO)
sobo_obs = observations.TestSOBO()

c = controller.Controller('test.json', interface = interface.TestInterface())

opt_params = c.get_named_parameters(['X1', 'X2'])


#MOBO test
mobo_obs = observations.TestMOBO()

samples = np.random.rand(10,2)
sampler = sample.Sample(opt_params, mobo_obs.children, c, samples)
sampler.run(10,1)
print(c.data)


#objs = [sobo_obs, mobo_obs.children[1]]
objs = mobo_obs.children[:-1]
ref = torch.tensor((1.0,5.0))
opt = mobo.MultiObjectiveBayesian(opt_params, objs, c, ref) 
opt.run(20,1)

#print(opt.gp.state_dict())


#plot to test
opt_data = c.group_data().loc[:,['1', '2']].to_numpy()
fig,ax = plt.subplots()
ax.plot(opt_data[20:,0],opt_data[20:,1],'+')
#c.group_data().plot('FocusingSolenoid', 'BuckingSolenoid')
plt.show()
