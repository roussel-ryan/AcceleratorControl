import numpy as np
import logging
import torch
import controller
import observations

def main():
    logging.basicConfig(level=logging.INFO)
    obs = observations.Test()

    c = controller.AWAController('test.json', testing = True)

    c.set_parameters(torch.tensor((20.0,10.0)),['FocusingSolenoid', 'BuckingSolenoid'])
    c.do_scan('FocusingSolenoid', obs)
    print(c.observation_data)

main()
