# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:25:57 2021

@author: awa
"""

import pandas as pd

data = pd.read_pickle('data/data_1617914840.pkl')
print(data.keys())
print(data['EMIT'].corr(data['DQ5']))
print(data['EMIT'].corr(data['FocusingSolenoid']))
print(data['EMIT'].corr(data['MatchingSolenoid']))

data.hist('DQ5')
data.hist('MatchingSolenoid')
data.hist('FocusingSolenoid')
data.hist('EMIT')