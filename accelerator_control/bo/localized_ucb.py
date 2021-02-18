import torch
from botorch.acquisition.multi_objective.analytic import MultiObjectiveAnalyticAcquisitionFunction
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement


class LocalEHVI(MultiObjectiveAnalyticAcquisitionFunction):
    def __init__(model, ref_point, partitioning, precision):
        super().__init__(

