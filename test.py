import numpy as np
import logging
import torch
import controller
import observations
import optimization

def main():
    logging.basicConfig(level=logging.INFO)
    obs = observations.Test()
    save_image = observations.ImageSave('pics')
    
    c = controller.AWAController('test.json', testing = True)

    c.set_parameters(torch.tensor((20.0,10.0)),['FocusingSolenoid', 'BuckingSolenoid'])
    #c.do_scan('FocusingSolenoid', obs)
    c.observe(save_image, 5, wait_time = 1)

    #opt_params = c.get_parameters(['FocusingSolenoid','BuckingSolenoid'])
    #opt_obj = obs
    #opt = optimization.SingleObjectiveBayesian(opt_params, opt_obj, c)

    #opt.optimize()

    #print(c.observation_data)
    
main()
