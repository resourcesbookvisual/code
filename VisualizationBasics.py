# -*- coding: utf-8 -*-

#%%

import pandas as pd

#link to data
linkRepo='https://github.com/resourcesbookvisual/data/'
linkFile='raw/master/eduwa.csv'
fullLink=linkRepo+linkFile
#
#getting the data:
eduwa=pd.read_csv(fullLink)
#
#what you have
eduwa.dtypes

