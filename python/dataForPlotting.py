#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:19:38 2021

@author: JoseManuel
"""

#%%

# link to data
linkRepo ='https://github.com/resourcesbookvisual/data/'
linkDemo ="raw/master/eduwa.csv " 

# activating Pandas and getting the data:
import pandas as pd
eduwa = pd.read_csv(linkRepo + linkDemo)

#what you have
eduwa.dtypes

#%%