import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from botorch.acquisition.multi_objective.analytic import MultiObjectiveAnalyticAcquisitionFunction
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement


class LocalizedEHVI(ExpectedHypervolumeImprovement):
    def __init__(self, model, ref_point, partitioning, precision):
        super().__init__(model, ref_point, partitioning)
        self.register_buffer('precision', precision)

        #print(dir(self.model))
        
    def forward(self, X):
        EHVI = super().forward(X)
        #print(f'EHVI {EHVI.shape}')
        #multiply EHVI by Gaussian centered at last point
        last_pt = self.model.train_inputs[0][0][-1].double()
        
        #define multivariate normal
        d = MultivariateNormal(last_pt, self.precision.double())

        #use pdf to calculate the weighting at the new point
        weight = torch.exp(d.log_prob(X).flatten()) / torch.exp(d.log_prob(last_pt).flatten())
        print(weight)
        #print(EHVI * weight)
        #print(f'weight {weight.shape}')
        
        return EHVI * weight
        
        
        

