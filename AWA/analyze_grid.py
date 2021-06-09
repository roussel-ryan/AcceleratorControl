import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append('\\'.join(os.getcwd().split('\\')[:-1]))

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from accelerator_control import transformer

data = pd.read_pickle('data/2d_scan_big.pkl')

#print(data.loc[data['MatchingSolenoid'] < 1.0].loc[data['DQ5'] > 0.2].head(40))

# print(data.corr())

data.plot('MatchingSolenoid', 'FocusingSolenoid', style='.')

fig = plt.figure()
ax = fig.add_subplot()


alg_samples = data
bad_samples = alg_samples[alg_samples['IMGF'] == 0]
good_samples = alg_samples[alg_samples['IMGF'] == 1]
samples = [bad_samples, good_samples]
labels = ['Not Successful', 'Successful']

for i in [0, 1]:
    sub_data = samples[i]
    ax.scatter(sub_data['MatchingSolenoid'],
               sub_data['FocusingSolenoid'], marker='o', s=20, label=labels[i])
ax.legend()
ax.set_xlabel('MatchingSolenoid')
ax.set_ylabel('FocusingSolenoid')

fig2, ax2 = plt.subplots()
ax2.plot(data['MatchingSolenoid'], data['EMIT'], '.')

safe_data = data.dropna()
X = safe_data[['MatchingSolenoid', 'FocusingSolenoid']].to_numpy()
XC = data[['MatchingSolenoid', 'FocusingSolenoid']].to_numpy()
F = safe_data['EMIT'].to_numpy().reshape(-1, 1)

print(X)

trans_x = transformer.Transformer(X, 'normalize')
trans_f = transformer.Transformer(F, 'standardize')

X = torch.from_numpy(trans_x.forward(X))
F = torch.from_numpy(trans_f.forward(F))
XC = torch.from_numpy(XC)
C = torch.from_numpy(data['IMGF'].to_numpy().reshape(-1, 1))

gp = SingleTaskGP(X, F)
cgp = SingleTaskGP(XC, C)

mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

cmll = ExactMarginalLogLikelihood(cgp.likelihood, cgp)
fit_gpytorch_model(cmll)

print(gp.covar_module.base_kernel.lengthscale)

n = 20
x = np.linspace(0, 1, n)
xx = np.meshgrid(x, x)
pts = np.vstack([ele.ravel() for ele in xx]).T
pts = np.hstack(pts)
pts = torch.from_numpy(pts)

with torch.no_grad():
    post = gp.posterior(pts)
    mean = post.mean
    var = torch.sqrt(post.variance)

    cpost = cgp.posterior(pts)
    cmean = cpost.mean
    cvar = torch.sqrt(post.variance)

fig3, ax3 = plt.subplots(1, 2)
mean = trans_f.backward(mean.numpy())
var = trans_f.backward(var.numpy()) - trans_f.means
pts = trans_x.backward(pts.numpy())

c1 = ax3[0].pcolor(pts.T[0].reshape(n, n),
                   pts.T[1].reshape(n, n),
                   mean.reshape(n, n))
c2 = ax3[1].pcolor(pts.T[0].reshape(n, n),
                   pts.T[1].reshape(n, n),
                   var.reshape(n, n))

ax3[0].set_title('Mean')
ax3[1].set_title('Uncertaintity')

ax3[0].set_xlabel('MatchingSolenoid')
ax3[1].set_xlabel('MatchingSolenoid')
ax3[0].set_ylabel('FocusingSolenoid')
ax3[1].set_ylabel('FocusingSolenoid')

fig3.colorbar(c1, ax=ax3[0])
fig3.colorbar(c2, ax=ax3[1])
plt.show()
