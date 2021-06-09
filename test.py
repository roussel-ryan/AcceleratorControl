import numpy as np
import logging
import torch
from accelerator_control import controller, observations, interface
from accelerator_control.algorithms import explore

import matplotlib.pyplot as plt

# main()
logging.basicConfig(level=logging.DEBUG)
sobo_obs = observations.TestSOBO()


c = controller.Controller('test.json', controller_interface=interface.TestInterface())

opt_params = c.get_named_parameters(['FocusingSolenoid', 'BuckingSolenoid'])

# MOBO test
mobo_obs = observations.TestMOBO(n_samples=5)
for i in range(20):
    c.set_parameters(opt_params, np.random.rand(2))
    # c.observe(sobo_obs)
    c.observe(mobo_obs)

print(c.data)

objs = mobo_obs.children[:2]
constraints = [mobo_obs.children[0], mobo_obs.children[-1]]
ref = torch.tensor((1.0, 5.0))
opt = explore.BayesianExploration(opt_params, objs, c, constraints, n_steps=10)
opt.run()
print(c.data)

# print(opt.gp.state_dict())


# plot to test
opt_data = c.group_data().loc[:, ['FocusingSolenoid', 'BuckingSolenoid']].to_numpy()
fig, ax = plt.subplots()
ax.plot(opt_data[20:, 0], opt_data[20:, 1])
# c.group_data().plot('FocusingSolenoid', 'BuckingSolenoid')
plt.show()
