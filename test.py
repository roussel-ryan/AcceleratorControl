import numpy as np
import logging
import torch
from accelerator_control import controller, observations, sobo, mobo, interface

import matplotlib.pyplot as plt

def main():
    logging.basicConfig(level=logging.INFO)
    obs = observations.Observation("obs1")
    image_save = observations.ImageSave('pics')

    #define interface and controller
    interf = interface.AWAInterface(True, True)
    c = controller.AWAController('test.json', interf)

    c.set_parameters(torch.Tensor(np.array((3,4))),['FocusingSolenoid', 'BuckingSolenoid'])
    #c.do_scan('FocusingSolenoid', 20, 50, obs)
    c.observe(obs, 5)

    opt_params = c.parameters
    opt_obj = obs
    opt = optimization.SingleObjectiveBayesian(opt_params, opt_obj, c)
    opt.optimize()
    #opt.optimize()

    print(c.data)
    
#main()
logging.basicConfig(level=logging.INFO)
sobo_obs = observations.TestSOBO()
#obs = observations.AWABeamSize()
#image_save = observations.ImageSave('pics')

c = controller.AWAController('test.json',True)

opt_params = c.get_parameters(['FocusingSolenoid', 'BuckingSolenoid'])

#c.do_scan('FocusingSolenoid', 20, 50, obs)


#SOBO test
#opt_obj = obs
#c.observe(obs, 2)
#opt = sobo.SingleObjectiveBayesian(opt_params, opt_obj, c, beta = 2.0)

#MOBO test
mobo_obs = observations.TestMOBO()
for i in range(10):
    c.set_parameters(opt_params, np.random.rand(2))
    c.observe(sobo_obs)
    c.observe(mobo_obs)

print(c.data)

objs = [sobo_obs, mobo_obs.children[0]]
opt = mobo.MultiObjectiveBayesian(opt_params, objs, c) 
opt.optimize(35,1)
print(c.data)


#plot to test
c.group_data().plot('TestMOBO.1','TestSOBO','scatter')
plt.show()
