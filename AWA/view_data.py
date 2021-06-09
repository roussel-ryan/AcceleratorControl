# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:25:57 2021

@author: awa
"""

import pandas as pd
import h5py
import matplotlib.pyplot as plt

data = pd.read_pickle('data/2d_scan_quad_adaptave.pkl')
print(data.keys())
print(data['EMIT'].corr(data['DQ5']))
print(data['EMIT'].corr(data['FocusingSolenoid']))
print(data['EMIT'].corr(data['MatchingSolenoid']))

#data.plot('DQ5','EMIT',style='.')
#data.plot('DQ5','IXXI',style='.')
#data.plot('DQ5','IXPXPI',style='.')
#data.plot('DQ5','IXXPI',style='.')

data.plot('FocusingSolenoid','MatchingSolenoid', style='.')

data.hist('DQ5')
data.hist('DQ4')

data.hist('MatchingSolenoid')
data.hist('FocusingSolenoid')
data.hist('EMIT')

