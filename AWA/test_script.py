import sys, os
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)

from accelerator_control import controller, pre_observation
from accelerator_control.algorithms import sample, explore
import interface
import observations
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)



def main():
    logging.debug('test')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    #define controller
    interf = interface.AWAInterface(False, Testing = False)
    c = controller.Controller('config.json', interface = interf)

    screen_obs = observations.AWAScreen(n_samples = 5)
    
    scan_vals = np.array((6.05,1.55)).reshape(1,-1)
    #scan_vals = np.random.rand(1,2)
    
    param = c.get_named_parameters(['FocusingSolenoid','MatchingSolenoid'])
    pre_observation_func = pre_observation.Sleep(5)
    #pre_observation_func = pre_observation.KeyPress()
   
    
    sampling = sample.Sample(param,
                            [screen_obs],
                             c, scan_vals,
                             normalized = False,
                             pre_observation_function = pre_observation_func)
    sampling.run()
    
    
    
    #do bayesian exploration
    roi_constraint = screen_obs.children[9]
    bexp = explore.BayesianExploration(param,
                                       screen_obs.children[:2],
                                       c,
                                       [roi_constraint],
                                       n_steps = 40,
                                       sigma = 0.1)
    bexp.run()
    #c.load_data('data/good_exploration_3.pkl')
    print(c.data[['MatchingSolenoid','FocusingSolenoid',
                  'FWHMX','FWHMY','IMGF']])
    model = bexp.create_model()
    bexp.get_acq(model, True)
    

if __name__ == '__main__':
    
    main()  
    
