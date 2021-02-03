import numpy as np

'''lightweight class for normalization using numpy'''


class Transformer:
    def __init__(self, x, transform_type = 'normalize'):
        possible_transformations = ['normalize','standardize']
        assert transform_type in possible_transformations
        assert len(x.shape) == 2

        
        self.ttype = transform_type
        self.x = x

        self._get_stats()
        
    def _get_stats(self):
        if self.ttype == 'normalize':
            self.mins = np.min(self.x, axis = 0)
            self.maxs = np.max(self.x, axis = 0) 

        elif self.ttype == 'standardize':
            self.means = np.mean(self.x, axis = 0)
            self.stds = np.std(self.x, axis = 0)
            
    def forward(self, x_old):
        x = x_old.copy()
        assert len(x.shape) == 2
        if self.ttype == 'normalize':
            for i in range(x.shape[1]):
                x[:,i] = (x[:,i] - self.mins[i]) /(self.maxs[i] - self.mins[i])

        elif self.ttype == 'standardize':
            for i in range(x.shape[1]):
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