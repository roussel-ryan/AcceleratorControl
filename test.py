import numpy as np
import logging
import torch
import controller
import observations
import sobo

def main():
    logging.basicConfig(level=logging.INFO)
    obs = observations.Observation("obs1")
    image_save = observations.ImageSave('pics')
    
    c = controller.AWAController('test.json',Testing=True)

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
obs = observations.Observation("obs1")
image_save = observations.ImageSave('pics')

c = controller.AWAController('test.json',True)

opt_params = c.get_parameters(['FocusingSolenoid', 'BuckingSolenoid'])
c.set_parameters(opt_params, np.array((3,4.0)))
#c.do_scan('FocusingSolenoid', 20, 50, obs)
c.observe(obs, 2)

opt_obj = obs
opt = sobo.SingleObjectiveBayesian(opt_params, opt_obj, c, beta = 2.0)
opt.optimize(15,1)
#opt.optimize()

print(c.data)
