import numpy as np

'''lightweight class for normalization using numpy'''


class Transformer:
    def __init__(self, x, transform_type = 'normalize'):
        possible_transformations = ['unitary','normalize','standardize']
        assert transform_type in possible_transformations
        assert len(x.shape) == 2

        
        self.ttype = transform_type
        self.x = x

        self._get_stats()
        
    def _get_stats(self):
        #note ignore nans
        
        if self.ttype == 'normalize':
            self.mins = np.nanmin(self.x, axis = 0)
            self.maxs = np.nanmax(self.x, axis = 0) 

        elif self.ttype == 'standardize':
            self.means = np.nanmean(self.x, axis = 0)
            self.stds = np.nanstd(self.x, axis = 0)

    def recalculate(self, x):
        #change transformer data and recalculate stats
        self.x = x
        self._get_stats()
    
    def forward(self, x_old):
        x = x_old.copy()
        assert len(x.shape) == 2

        
        if self.ttype == 'normalize':
            for i in range(x.shape[1]):
                if self.maxs[i] - self.mins[i] == 0.0:
                    x[:,i] = x[:,i] - self.mins[i]
                else:
                    x[:,i] = (x[:,i] - self.mins[i]) /(self.maxs[i] - self.mins[i])
                    
        elif self.ttype == 'standardize':
            for i in range(x.shape[1]):
                if self.stds[i] == 0:
                    x[:,i] = x[:,i] - self.means[i]
                else:
                    x[:,i] = (x[:,i] - self.means[i]) / self.stds[i]

        return x
                
    def backward(self, x_old):
        x = x_old.copy()
        assert len(x.shape) == 2
        
        if self.ttype == 'normalize':
            for i in range(x.shape[1]):
                x[:,i] = x[:,i] * (self.maxs[i] - self.mins[i]) + self.mins[i]

        elif self.ttype == 'standardize':
            for i in range(x.shape[1]):
                x[:,i] = x[:,i] * self.stds[i] + self.means[i]

        return x
                
if __name__ == '__main__':
    #testing suite
    x = np.random.uniform(size = (10,2)) * 10.0
    print(x)
    t = Transformer(x, 'standardize')
    x_new = t.forward(x)
    print(x_new)
    print(t.backward(t.forward(x)))
