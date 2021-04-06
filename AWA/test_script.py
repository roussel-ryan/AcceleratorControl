import sys, os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from accelerator_control import controller, pre_observation
from accelerator_control.algorithms import sample
import interface
import observations
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)

def main():
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    #define controller
    interf = interface.AWAInterface(Testing = False)
    c = controller.Controller('config.json', interface = interf)

    screen_obs = observations.AWAScreen(n_samples = 20, image_directory = 'pics/')
    
    scan_vals = np.linspace(1, 2.5, 10).reshape(-1,1)
    
    param = c.get_named_parameters(['MatchingSolenoid'])
    pre_observation_func = pre_observation.Sleep(3)
    #pre_observation_func = pre_observation.KeyPress()
   
    #sampling = sample.Sample(param,
    #                         [screen_obs],
    #                         c, scan_vals, 
    #                         pre_observation_function = pre_observation_func)
    #sampling.run()
    
    
    c.set_parameters(param,
                     np.array([1.6]))
    
    c.observe(screen_obs)
    #c.set_parameters(param, np.random.rand(1,1))

    #c.observe(screen_obs)
    print(c.data)
    

if __name__ == '__main__':
    main()  
    
