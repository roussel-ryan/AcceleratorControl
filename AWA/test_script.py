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
    logging.getLogger('accelerator_control.algorithms').setLevel(logging.INFO)
    logging.getLogger('emittance_calculation').setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    #define controller
    interf = interface.AWAInterface(False, Testing = False)
    c = controller.Controller('config.json', interface = interf)

    screen_obs = observations.AWAScreen(n_samples = 5)
    emittance_obs = observations.Emittance(0.002, 2.84, 1086, n_samples = 5)
    
    #scan_vals = np.array(((8.3, 1.2),
    #                      (8.3, 1.4),
    #                      (8.3, 1.6)))
    scan_vals = np.array(((7.2, 1.5))).reshape(1,-1)
    #scan_vals = np.random.rand(2,2)
    
    param = c.get_named_parameters(['FocusingSolenoid','MatchingSolenoid'])
    pre_observation_func = pre_observation.Sleep(5)
    #pre_observation_func = pre_observation.KeyPress()
   
    if 0:
        sampling = sample.Sample(param,
                                 [screen_obs],
                                 c, scan_vals,
                                 normalized = False,
                                 pre_observation_function = pre_observation_func)
        sampling.run()
    
    emit_sampling = sample.Sample(param,
                            [emittance_obs],
                             c, scan_vals,
                             normalized = False,
                             pre_observation_function = pre_observation_func)
    emit_sampling.run()
    
    
    
    #do bayesian exploration
    if 0:
        roi_constraint = screen_obs.children[9]
        bexp = explore.BayesianExploration(param,
                                           [emittance_obs],
                                           c,
                                           [roi_constraint],
                                           n_steps = 20,
                                           sigma = 1.0)
        bexp.run()
        model = bexp.create_model()
        bexp.get_acq(model, True)
    
    #c.load_data('data/good_exploration_1_YAG6.pkl')
    print(c.data[['MatchingSolenoid','FocusingSolenoid',
                  'FWHMX','FWHMY','EMIT','IMGF','ROT_ANG']])
    

if __name__ == '__main__':
    main()  
    
