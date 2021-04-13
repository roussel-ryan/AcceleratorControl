import sys, os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from accelerator_control import controller
import interface
import observations
import numpy as np

import logging

def main():
    '''
    Simple testing script to test image processing
    This script does the following
    - estabishes a connection to the AWA control system
    - generates 5 random valid points to set the focusing solenoid strength 
        (modify config.json to set max and min setpoints 
         and the communication channel)
    - does 5 screen grabs and saves the data/screen images

    '''

    
    logging.basicConfig(level=logging.INFO)

    #define controller
    interf = interface.AWAInterface(testing= False)
    c = controller.Controller('config.json', controller_interface= interf)

    screen_obs = observations.AWAScreen(image_directory = 'pics')

    param_names = ['FocusingSolenoid']
    n_params = len(param_names)
    
    obj_params = c.get_named_parameters(param_names)

    #generate some random valid points
    n_pts = 5
    initial_pts = np.empty((n_pts,n_params))
    for i in range(n_params):
        initial_pts[:,i] = np.random.uniform(*obj_params[i].bounds,(n_pts,))

    #sample initial points
    for pt in initial_pts:
        c.set_parameters(obj_params, pt)
        c.observe(screen_obs, 5)

    print(c.data)
    
main()
    
