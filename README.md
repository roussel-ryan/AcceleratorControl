# Accelerator Control
Control system code for accelerators

Code is organized as follows
----------------------------

- accelerator_interface.py - low level I/O class for interacting with windows accelerator control software, contains methods to establish connections, set parameters and read image files

- controller.py - high level I/O class for controlling the accelerator, tracks setpoints, parameters and parameter channels, performs batch observations, and stores observed data; controller takes a json file that describes accelerator parameter names, channels, and setpoint bounds as well as observation parameters (screen coordinates, wait time between observations, number of observations to take)

- image_processing.py - methods for doing image processing, including thresholding, masking, filtering, and fitting

- observation.py - defines observation class, which contains the method for doing a given observation, ie. RMS beam size etc.
  - observations objects come in two flavors, Observation and GroupObservation -> see docstrings for details
- parameter.py - defines Parameter container class which stores info (channel, bounds, name) for each accelerator input parameter

- utilities.py - contains helper functions for normalization etc.
