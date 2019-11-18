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


#%%
# from R:
# table(eduwa$LocaleType,exclude = 'nothing')


eduwa.LocaleType.value_counts(dropna=False)

#%%
