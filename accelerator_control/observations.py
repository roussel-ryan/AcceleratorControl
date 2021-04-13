import numpy as np
import pandas as pd


# import image_processing as ip


class Observation:
    """
    Observation function class that keeps track of function name and callable
    function to do the observation.

    If this observation can be made simultaneously with other observations_list
    using the same process pass it to a GroupObservation.add_child().
    By default a observation has no parent =(.

    An example of a simultaneous observation is getting the x and y beamsize
    from a single image.

    If is_child is true, calls to this observation will execute the
    group callable method instead.

    This is done so that optimizers can call individual observations_list while
    still passively collecting excess data during optimization.

    For use, overwrite __call__().
    Observation callables return a pandas DataFrame object with
    column names = to observation name.

    Arguments
    ---------
    name : string

    """

    def __init__(self, name, n_samples=1):
        self.name = name
        self.is_child = False
        self.observation_index = 0
        self.n_samples = n_samples
        self.parent = None

    def add_parent(self, parent):
        """adds parent to observation -> self._call__ is not necessary"""

        self.is_child = True
        self.parent = parent

    def __call__(self, controller, param_dict):
        """
        do observation

        Arguments
        ---------
        controller : controller.Controller object

        """
        if self.is_child:
            return self.parent()
        else:
            raise NotImplementedError


class GroupObservation:
    """
    group class for when multiple observations_list can be made simultaneously
    (for example multiple aspects of the beam can be measured at once w/ an image)
    - when a child observation is called the parent observation is called instead
    - child observation name is <parent name>.<child name>

    """

    def __init__(self, name, output_names, n_samples=1):
        self.name = name
        self.children = []
        self.n_samples = n_samples
        self.is_child = False
        self.observation_index = 0

        self.output_names = output_names

        # add children observations_list
        for name in self.output_names:
            obs = Observation(name)
            self.add_child(obs)

    def __call__(self, controller, param_dict):
        """

        impliments observation procedure,
        should return a pandas DataFrame object

        """
        raise NotImplementedError

    def add_child(self, child):
        child.add_parent(self)
        self.children += [child]

    def get_children_names(self):
        return [child.name for child in self.children]


class TestSOBO(Observation):
    """observation class used for testing SOBO algorithm"""

    def __init__(self, name='TestSOBO'):
        super().__init__(name)

    def __call__(self, controller, param_dict):
        val = controller.interface.test_observation()
        param_dict[self.name] = np.ones(self.n_samples) * val[0]
        return pd.DataFrame(param_dict, index=[0])


class TestMOBO(GroupObservation):
    """
    observation class used for testing MOBO algorithm
    '1' and '2' return ZDT1 values
    '3' returns 0 if x[0] < 0.5, 1 otherwise

    """

    def __init__(self, name='TestMOBO', n_samples=1):
        output_names = ['1', '2', '3']
        super().__init__(name, output_names, n_samples)

    def __call__(self, controller, param_dict):
        vals = controller.interface.test_observation(self.n_samples)
        for i in range(len(self.output_names)):
            param_dict[self.output_names[i]] = vals.T[i]

        return pd.DataFrame(param_dict)
