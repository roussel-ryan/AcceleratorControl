from accelerator_control import controller, interface, observations, mobo
import numpy as np

def main():
    #view_params()
    #define controller
    interf = interface.SLACInterface()
    c = controller.Controller('config.json', interf)

    #get initial setpoint (set to minimum bound for each parameter as example
    initial = np.zeros(len(c.parameters))
    for i in range(len(initial)):
        initial[i] = c.parameters[i].bounds[0]

    print([param.name for param in c.parameters])
    print(initial)
        
    c.set_parameters(c.parameters, initial)
    
    xy_observation = observations.OTR2Profiles(measure_z = False)
    xyz_observation = observations.OTR2Profiles(measure_z = True)
    
    #do MOBO
    mobo_objectives = [xyz_observation.children[0],
                       xyz_observation.children[2]]
    #mobo params
    param_names = ['SOL1:solenoid_field_scale',
                   'CQ01:b1_gradient',
                   'SQ01:b1_gradient']
    obj_params = c.get_named_parameters(param_names)

    #generate some random valid points
    n_pts = 10
    initial_pts = np.empty((n_pts,3))
    for i in range(3):
        initial_pts[:,i] = np.random.uniform(*obj_params[i].bounds,(10,))

    #sample initial points
    for pt in initial_pts:
        c.set_parameters(obj_params, pt)
        c.observe(xyz_observation, 5)

    print(c.data)
        
    opt = mobo.MultiObjectiveBayesian(obj_params, mobo_objectives, c)
    opt.optimize(20,1)
    print(c.data[param_names + ['sigma_x','sigma_z']])

main()
