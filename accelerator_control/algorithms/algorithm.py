import numpy as np
import torch
import logging
import time
import os

from .. import observations
from .. import parameter
from .. import transformer


class Algorithm:
    """
    Algorithm class for use with accelerator operations
    Must implement acquire_point() function.
    To do something model assisted implement create_model() function.

    """

    def __init__(self, parameters, observations_list, controller,
                 **kwargs):
        """
        Initialize algorithm

        Arguments
        ---------
        parameters : list of parameters
             List of parameter.Parameter objects used for optimization

        observations_list : list of observations.Observation
             Observation objects needed

        controller : controller.Controller
             Controller object used to control the accelerator

        pre_observation_function : callable, optional
             Optional function called between parameter changes and
             performing observations. Default is None

        """
        assert isinstance(parameters[0], parameter.Parameter)

        self.logger = logging.getLogger(__name__)

        self.parameters = parameters
        self.observations = observations_list
        self.controller = controller

        self.parameter_names = parameters
        self.n_parameters = len(self.parameters)
        self.n_observations = len(self.observations)
        self.n_steps = kwargs.get('n_steps', 10)
        self.pre_observation_function = kwargs.get('pre_observation_function', None)
        self.save_images = kwargs.get('save_data', False)

        if self.save_images:
            directory = f'pics/run_{time.time:d}'
            os.mkdir(directory)
            for ele in self.observations:
                ele.image_directory = directory

    def create_model(self):
        """
        create a model to be passed to an acquisition function
        else return None

        Use self.get_data() to get data for model
        """
        return None

    def acquire_point(self, model=None):
        """
        Algorithm used to pick the next observation point
        should return a NORMALIZED CANDIDATE [0,1)
        """

        raise NotImplementedError

    def get_data(self, normalize_x=True, normalize_f=True):
        """
        return numpy array with observations_list and convert to torch
        Note: by default input parameters are normalized from 0 to 1
        objective values are standardized to zero mean and unit variance

        Always use normalize = True unless doing visualization!!!

        """

        # data = self.controller.data.groupby(['state_idx']).mean()
        data = self.controller.data

        logging.debug(f'observations_list list {[obs.name for obs in self.observations]}')
        f = data[[obj.name for obj in self.observations]]
        f = f.to_numpy()
        x = data[[p.name for p in self.parameters]].to_numpy()

        self.logger.debug(f'Raw data from controller:\nX:\n{x}\nf:\n{f}')

        # check if we accidentally forgot to specify that a column is a constraint
        for i in range(f.shape[1]):
            if np.array_equal(np.unique(f[:, i]), np.array((0.0, 1.0))) and np.any(normalize_f[i]):
                self.logger.warning(f'data column {i} has only ones and zeros' +
                                    ' but normalization flag is set to 1!' +
                                    ' Did you mean for this column to be normalized?')

        # by default normalize all f (set all of normalize flags to 1)
        if np.all(normalize_f):
            f_nflags = np.ones_like(f[0])
        else:
            assert isinstance(normalize_f, np.ndarray)
            f_nflags = normalize_f

        logging.debug(f'normalization fflags {f_nflags}')
        f_normed = self._apply_f_normalization(f, f_nflags)

        # by default normalize all x (set all of normalize flags to 1)
        if np.all(normalize_x):
            x_nflags = np.ones_like(x[0])
        else:
            assert isinstance(normalize_x, np.ndarray)
            x_nflags = normalize_x

        x_normed = self._apply_x_normalization(x, x_nflags)

        x = torch.from_numpy(x_normed)
        f = torch.from_numpy(f_normed)

        self.logger.debug(f'Scaled data from controller:\nX:\n{x}\nf:\n{f}')
        return x, f

    def run(self):
        """
        run the algorithm
        """
        self.logger.info(f'Starting algorithm run with {self.n_steps} steps')

        for i in range(self.n_steps):
            self.logger.info(f'Step {i}')

            self.logger.debug('creating model')
            model = self.create_model()

            # acquire the next point to observe - in normalized space
            self.logger.debug('acquiring next point')
            tstart = time.time()
            candidate = self.acquire_point(model)
            self.logger.info(f'acquired point in {time.time() - tstart:.2f} s')
            self.logger.debug(f'normalized candidate is {candidate}')

            # un-normalize candidate
            unn_c = np.zeros_like(candidate)
            for j in range(self.n_parameters):
                unn_c[0][j] = self.parameters[j].transformer.backward(
                    candidate[0][j].numpy().reshape(1, 1))
            self.logger.debug(f'un-normed candidate is {unn_c}')

            # set parameters
            self.logger.debug('setting parameters')
            self.controller.set_parameters(self.parameters,
                                           unn_c.flatten().astype(
                                               np.float32))

            # execute pre-observation function
            # self.logger.info(self.pre_observation_function)
            # if not self.pre_observation_function == None:
            # self.pre_observation_function(self.controller)
            time.sleep(1.5)

            # make observations_list necessary (possibly grouped observations_list)
            # the majority of measurements must be made in serial
            self.logger.info('starting observations_list')
            required_observations = []
            for obs in self.observations:
                # check to see if an observation is a child
                if obs.is_child:
                    # check to make sure this observation is not in any of
                    # the other parents
                    expanded_required_observations = []
                    for sub_obs in required_observations:
                        if isinstance(sub_obs, observations.GroupObservation):
                            expanded_required_observations += sub_obs.children

                        else:
                            expanded_required_observations += sub_obs
                    exp_obs_names = [ele.name for ele in expanded_required_observations]
                    if obs.name not in exp_obs_names:
                        if obs.parent not in required_observations:
                            required_observations += [obs.parent]
                else:
                    required_observations += [obs]

            self.logger.info('doing observations_list ' +
                             f'{[obs.name for obs in required_observations]}')

            for obs in required_observations:
                self.controller.observe(obs)

            self.logger.info('observations_list done')

    def _apply_f_normalization(self, f, normalize_flags):
        """
        by default apply standardized normalization
        - to modify transformer type implement get_f_transformers(f)
        """

        try:
            self.f_transformers = self.get_f_transformers(f)

        except NotImplementedError:
            self.f_transformers = [
                transformer.Transformer(ele.reshape(-1, 1),
                                        'standardize') for ele in f.T]

        logging.debug(f'number of f transformers: {len(self.f_transformers)}')

        # normalize f according to reference point
        f_normed = f.copy()
        for i in range(self.n_observations):
            if normalize_flags[i]:
                f_normed[:, i] = self.f_transformers[i].forward(
                    f[:, i].reshape(-1, 1)).flatten()

        return f_normed

    def _apply_x_normalization(self, x, normalize_flags):
        """
        by default apply normalization based on input bounds

        """

        # normalize each input vector
        x_normed = x.copy()
        for i in range(self.n_parameters):
            if normalize_flags[i]:
                x_normed[:, i] = self.parameters[i].transformer.forward(
                    x[:, i].reshape(-1, 1)).flatten()

        return x_normed

    def get_f_transformers(self, f):
        raise NotImplementedError
