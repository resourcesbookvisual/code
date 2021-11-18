#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:19:38 2021

@author: JoseManuel
"""

#%%

# link to data
linkRepo ='https://github.com/resourcesbookvisual/data/'
linkEDU ="raw/master/eduwa.csv"
fullLink= linkRepo + linkEDU 

# activating Pandas and getting the data:
import pandas as pd
eduwa = pd.read_csv(fullLink)

#what you have
eduwa.dtypes

#%%