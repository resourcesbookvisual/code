# -*- coding: utf-8 -*-
"""
Python Codes for Chapter: 
Visualization Basics
"""

#%%
linkRepo='https://github.com/resourcesbookvisual/data/'
linkDemo="raw/master/demo.rds" # RDATA file!
fullLink=linkRepo+linkDemo

link="https://github.com/EvansDataScience/data/raw/master/crime.RData"

import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()
readRDS = robjects.r['load']
base = importr('base')
demo = readRDS(file=base.url(link))
#demo = pandas2ri.ri2py(demo)

#%%
demo = pandas2ri.ri2py(demo)

#%%
import cloudpickle as cp
from urllib.request import urlopen
pko = cp.load(urlopen('https://github.com/resourcesbookvisual/data/raw/master/demo.pkl'))

#%%
import pickle
from urllib.request import urlopen
linkRepo='https://github.com/resourcesbookvisual/data/'
linkDemo="raw/master/demo.pkl" # RDATA file!
fullLink=linkRepo+linkDemo

respk22 = pickle.load(urlopen(fullLink))
#%%
link="https://github.com/EvansDataScience/data/raw/master/crime.RData"

from numpy import *
import scipy as sp
from pandas import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

ro.r('load(link)')

#%%

from rpy2.robjects.packages import importr, data,oad
datasets = importr('datasets')
mtcars_env = data(datasets).fetch(fullLink)
mtcars = mtcars_env['crime']