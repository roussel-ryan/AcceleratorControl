import os
import sys

sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from accelerator_control import controller
from accelerator_control.algorithms import explore
import interface
import observations
import logging

logging.getLogger().setLevel(logging.INFO)


def main():
    logging.debug('test')

    # define controller
    interf = interface.SLACInterface()
    control = controller.Controller('config.json', controller_interface=interf)

    emittance_observation_group = observations.Emittance(n_samples=5)

    param_names = ['SOL1', 'CQ01', 'SQ01']
    params = control.get_named_parameters(param_names)

    # observes both the emittance and the validity of the measurement
    # based on upper/lower bounds
    control.observe(emittance_observation_group.children)

    # do bayesian exploration
    objectives = [emittance_observation_group.children[0]]
    constraints = [emittance_observation_group.children[1]]
    bexp = explore.BayesianExploration(params, objectives, control,
                                       constraints, n_steps=50, sigma=100)

    if 0:
        bexp.run()
        # model = bexp.create_model()
        # bexp.get_acq(model, True)

    # c.load_data('data/good_3_parameter_exploration.pkl')
    print(control.data[['EMIT', 'EMITF'] + param_names])
    # print(c.data)
    # model = bexp.create_model()
    # models = model.models
    # for m in models:
    #    print(m.covar_module.base_kernel.lengthscale)
    # bexp.get_acq(model, True)
    # bexp.plot_model(0, False)


if __name__ == '__main__':
    main()
