# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:25:57 2021

@author: awa
"""

import pandas as pd
import h5py
import matplotlib.pyplot as plt

data = pd.read_pickle('data/data_1618002663.pkl')
print(data.keys())
print(data['EMIT'].corr(data['DQ5']))
print(data['EMIT'].corr(data['FocusingSolenoid']))
print(data['EMIT'].corr(data['MatchingSolenoid']))

data.plot('DQ5','EMIT',style='.')
data.plot('DQ5','IXXI',style='.')
data.plot('DQ5','IXPXPI',style='.')
data.plot('DQ5','IXXPI',style='.')

#data.hist('DQ5')
#data.hist('MatchingSolenoid')
#data.hist('FocusingSolenoid')
#data.hist('EMIT')

