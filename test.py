import numpy as np
import logging
import torch
from accelerator_control import controller, observations, sobo, mobo, interface, parameter

import matplotlib.pyplot as plt

    
#main()
logging.basicConfig(level=logging.INFO)
sobo_obs = observations.TestSOBO()
#obs = observations.AWABeamSize()
#image_save = observations.ImageSave('pics')


c = controller.Controller('test.json', interface = interface.TestInterface())

opt_params = c.get_named_parameters(['FocusingSolenoid', 'BuckingSolenoid'])

#c.do_scan('FocusingSolenoid', 20, 50, obs)


#SOBO test
#opt_obj = obs
#c.observe(obs, 2)
#opt = sobo.SingleObjectiveBayesian(opt_params, opt_obj, c, beta = 2.0)

#MOBO test
mobo_obs = observations.TestMOBO()
for i in range(20):
    c.set_parameters(opt_params, np.random.rand(2))
    c.observe(sobo_obs)
    c.observe(mobo_obs)

print(c.data)

objs = [sobo_obs, mobo_obs.children[1]]
#objs = mobo_obs.children
ref = torch.tensor((1.0,5.0))
opt = mobo.MultiObjectiveBayesian(opt_params, objs, c, ref) 
opt.optimize(10,1)
print(c.data)

#print(opt.gp.state_dict())


#plot to test
opt_data = c.group_data().loc[:,['FocusingSolenoid', 'BuckingSolenoid']].to_numpy()
fig,ax = plt.subplots()
ax.plot(opt_data[20:,0],opt_data[20:,1])
#c.group_data().plot('FocusingSolenoid', 'BuckingSolenoid')
plt.show()
