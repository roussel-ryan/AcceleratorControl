import sys, os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from accelerator_control import controller
import interface
import observations
import numpy as np

import logging

def main():
    logging.basicConfig(level=logging.INFO)
    
    #define controller
    interf = interface.AWAInterface(Testing = True)
    c = controller.Controller('test_config.json', interface = interf)

    screen_obs = observations.AWAScreen(image_directory = 'pics')

    param = c.get_named_parameters(['test_param'])

    c.set_parameters(param, np.ones((1,1)))
    
    c.observe(screen_obs)
    print(c.data)
    
main()
    
