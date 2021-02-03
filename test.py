import numpy as np
import logging
import torch
import controller
import observations
import optimization

def main():
    logging.basicConfig(level=logging.INFO)
    obs = observations.Test()
    image_save = observations.ImageSave('pics')
    
    c = controller.AWAController('test.json', testing = True)

    c.set_parameters(np.array((10,50)),['FocusingSolenoid', 'BuckingSolenoid'])
    #c.do_scan('FocusingSolenoid', 20, 50, obs)
    c.observe(image_save, 5, wait_time = 1)

    #opt_params = c.get_parameters(['FocusingSolenoid','BuckingSolenoid'])
    #opt_obj = obs
    #opt = optimization.SingleObjectiveBayesian(opt_params, opt_obj, c)

    #opt.optimize()

    print(c.data)
    
main()
