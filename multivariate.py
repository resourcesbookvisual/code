#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:25:13 2020

@author: JoseManuel
"""

#%%
import pandas as pd

#link to data
linkRepo='https://github.com/resourcesbookvisual/data/'
linkFile='raw/master/crime.csv'
fullLink=linkRepo+linkFile
crime=pd.read_csv(fullLink)
# keeping years since 2008
crime2=crime[crime.year>=2008].copy()

# making data without missing values
crime2.dropna(subset=['DaysToReport'],inplace=True)
crime2.fillna(value={'crimecat': 'UNcategorized'},inplace=True)

# making table
crimeDate=pd.crosstab(crime2.OccurredDate,crime2.crimecat)
#%%
#table of dates to data frame
crimeDateDF=crimeDate.stack().reset_index()
#renaming columns
crimeDateDF.columns=['date','crime','count']
# formatting date column as Date type
crimeDateDF['date']=crimeDateDF['date'].astype('datetime64[ns]')

#%%
# sum by crimes, resulting in data frame:
ordCrime=crimeDateDF.groupby('crime').sum().reset_index()
# sum by crimes, resulting in data frame:
ordCrimeSort=ordCrime.sort_values(by=['count'],
                                  ascending=False)
#saving only names of crimes
descendCrimes=ordCrimeSort.crime.values
#setting the variable as an ordinal
crimeDateDF['crime']=pd.Categorical(crimeDateDF.crime, 
                                    ordered=True,
                                    categories=descendCrimes)

#%%
from plotnine import *

basetF = ggplot(crimeDateDF, aes(x='date',
                                 y='count')) +theme_classic()
basetF+= geom_point(alpha=0.1,shape='+') 

tspF= basetF + geom_smooth(fill='silver',
                           method='loess',
                           alpha=1,
                           color='white')

tspF += scale_x_datetime(date_labels='%Y',
                         date_breaks='2 years')

tspF += theme(axis_text_x  = element_text(angle=90,size=7),
              axis_text_y  = element_text(size=7),
              strip_text = element_text(size = 6))

tspF1 = tspF + facet_wrap('crime',
                          ncol=4,
                          scales="free_y") 

tspF1



#%%


tspF2 = tspF + facet_wrap('crime',
                          ncol=4) 

tspF2


#%%
import pandas as pd


#link to data
linkRepo='https://github.com/resourcesbookvisual/data'
linkSafe='/raw/master/safeCitiesIndexAll.csv'
fullLink=linkRepo+linkSafe
safe=pd.read_csv(fullLink)



#%%


safeAllLong=pd.melt(safe, id_vars=['city'])

safeAllLong.variable

#%%

positionsIN=safeAllLong.variable.str.contains('_In_')
safeIN=safeAllLong[positionsIN]


#%%


medVar=safeIN.groupby('variable').describe()['value'][['50%']]
varSorted=medVar.sort_values(by=['50%'],ascending=True).index

medCity=safeIN.groupby('city').describe()['value'][['50%']]
citySorted=medCity.sort_values(by=['50%'],ascending=True).index


#%%

#new labels for ticks
varLabel=[v[1] for v in varSorted.str.split('In_')]

# color for one city
colorCity=['r' if text=='Lima' else 'k' for text in citySorted]




#%%
from plotnine import *

#raw heatplot
base   = ggplot(safeIN, aes(x = 'variable',
                          y ='city')) 
heat1  = base +  geom_tile(aes(fill = 'value')) 

#reordering
heat2  = heat1 + scale_x_discrete(limits=varSorted,
                                 labels=varLabel)
heat2 += scale_y_discrete(limits=citySorted)
 
#change palette to highlight top, bottom and average
heat2 += scale_fill_gradient2(midpoint = 50,
                              mid= 'white',
                              low = 'red', 
                              high = 'darkgreen')
# Readable text
heat2 += labs(x="",y="")
heat2 += theme(axis_text_x = element_text(angle = 90, 
                                         va = 'top',
                                         size = 6),
               axis_text_y = element_text(size = 6))

# Highlighting one city 
import matplotlib.pyplot as plt

heat2FIG = heat2.draw() # plotnine to matplotlib
ax = heat2FIG.axes[0]   # the plotting area 
labels = ax.get_yticklabels() #the city labels
# for every current city label:
for l, c in zip(labels, colorCity):
    l.set_color(c) # change color




#%%
    
    
safe['meanDIN']=safe.filter(regex='D_In').mean(axis=1)
safe['meanDOUT']=safe.filter(regex='D_Out').mean(axis=1)

safe['meanHIN']=safe.filter(regex='H_In').mean(axis=1)
safe['meanHOUT']=safe.filter(regex='H_Out').mean(axis=1)

safe['meanIIN']=safe.filter(regex='I_In').mean(axis=1)
safe['meanIOUT']=safe.filter(regex='I_Out').mean(axis=1)

safe['meanPIN']=safe.filter(regex='P_In').mean(axis=1)
safe['meanPOUT']=safe.filter(regex='P_Out').mean(axis=1)

#%%


safeINS=safe.filter(regex='IN$|^city')
safeINS.columns=["city",'DIGITAL','HEALTH','INFRA','PERSON']


#%%

InValues=['DIGITAL','HEALTH','INFRA','PERSON']
safeINS['top']=safeINS.loc[::,InValues].mean(axis=1)>90

# To long version
safeINLongTop = pd.melt(safeINS, id_vars = ['city','top'])  

 

#%%

from plotnine import *
import numpy as np

conditionColor=np.where(safeINLongTop['top'],
                   'black',"silver")
conditionLabel=np.where(safeINLongTop['top'],
                   safeINLongTop['city'],"")
basep1 = ggplot(safeINLongTop, aes(x = 'variable',
                                 y = 'value', 
                                 group = 'city')) 
basep1 += theme_classic()
paral1  = basep1 +  geom_path(color=conditionColor) 
paral1+= geom_text(aes(label=conditionLabel))


paral1    






#%%223


#reordering
NewOrder=["DIGITAL", "INFRA", "PERSON", "HEALTH"]
safeINLongTop.variable=pd.Categorical(safeINLongTop.variable,
                                      categories=NewOrder,
                                      ordered=True)
safeINLongTop=safeINLongTop.sort_values(by=['variable'])


#%%




# reloading data frame 

conditionColor=np.where(safeINLongTop['top'],
                   'black',"silver")
conditionLabel=np.where(safeINLongTop['top'],
                   safeINLongTop['city'],"")
basep1b = ggplot(safeINLongTop, aes(x = 'variable',
                                   y = 'value', 
                                   group = 'city')) 
basep1b += theme_classic()
paral1b  = basep1b +  geom_path(color=conditionColor) 
paral1b += geom_text(aes(label=conditionLabel))







paral1b #258


#%%

from plotnine import *
import numpy as np

basep2 = ggplot(safeINLongTop[safeINLongTop.top],
                aes(x = 'variable',
                    y = 'value', 
                    group = 'city')) + theme_classic()

paral2 = basep2 +  geom_path(aes(color='city'))

paral2 += theme(legend_position="top",
                legend_title_align="center")

paral2




#%%

# NOT USED:
safeINS_long=pd.melt(safeINS, id_vars=['city'])
safeINS_long


#%%

cities=['Abu Dhabi', 'Lima' , 'Zurich','London']
# a copy (avoids changes to original)
safeRadarINS=safeINS.copy()
#index will be the city instead of usual numbers
safeRadarINS.index=safeRadarINS.city
#no need for column city, it is the index,
#and the column 'top' is deleted
safeRadarINS.drop(columns=['city','top'],inplace=True)
#choosing by index value with 'loc' (not 'iloc')
safeRadarINS=safeRadarINS.loc[cities,:]


safeRadarINS
#%%


# city Name - index.values[0]= ABU DHABI
currentName=safeRadarINS.index.values[0]
# variableNames 
VarNames=safeRadarINS.columns


# calling plotly
import plotly.graph_objects as go
from plotly.offline import plot

#%%

fig = go.Figure() # "fig" created, then add info:
fig.add_trace(go.Scatterpolar(
                        #data for each variable
                        r=safeRadarINS.loc[currentName],
                        #variable names
                        theta=VarNames))

fig.update_traces(fill='toself')#radar ends in the beginning
fig.update_layout(title = currentName)


#%%
##for Jupyter-like notebooks
fig.show()
 
##for Spyder-like environments
plot(fig, auto_open=True)

#%%
# libraries needed
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# making the grid
fig = make_subplots(rows=2, 
                    cols=2, 
                    specs=[[{'type': 'polar'}]*2]*2)

# the four subplots (traces)
currentName0=safeRadarINS.index.values[0]
fig.add_trace(go.Scatterpolar(
        name = currentName0,
        r = safeRadarINS.loc[currentName0],
        theta = VarNames),
        1, 1) # location of plot (row,columm)

currentName1=safeRadarINS.index.values[1]
fig.add_trace(go.Scatterpolar(
        name = currentName1,
        r = safeRadarINS.loc[currentName1],
        theta = VarNames),
        1, 2)# location of plot 

currentName2=safeRadarINS.index.values[2]
fig.add_trace(go.Scatterpolar(
        name = currentName2,
        r = safeRadarINS.loc[currentName2],
        theta = VarNames),
        2, 1)# location of plot 

currentName3=safeRadarINS.index.values[3]
fig.add_trace(go.Scatterpolar(
        name = currentName3,
        r = safeRadarINS.loc[currentName3],
        theta = VarNames),
        2, 2)# location of plot 

fig.update_traces(fill='toself')

#customization to show 0-100 range
layoutTrace={'radialaxis' : {'range': [0, 100]}}
fig.update_layout(polar1 = layoutTrace,
                  polar2 = layoutTrace,
                  polar3 = layoutTrace,
                  polar4 = layoutTrace,
                  showlegend=False)

##for Spyder-like environments
plot(fig, auto_open=True)



#%%

# libraries needed
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot

# variableNames 
VarNames=safeRadarINS.columns
# cityNames
CaseNames=safeRadarINS.index.values

# making up my theme_minimal
layoutTrace={'radialaxis':{'visible':True,
                           'linecolor': 'black',
                           'gridcolor': 'silver',
                           'range' :[0, 100]},
             'angularaxis': {'gridcolor': 'silver',
                             'linecolor':'black'},
             'bgcolor': 'white'} #background
           
#number of rows and columns
nR=2;nC=2 
           
           
# producing figure as collection of subplots
fig = make_subplots(rows=nR, cols=nC, #dimensions
                    specs=[[{'type': 'polar'}]*nC]*nR,
                    subplot_titles=CaseNames) #city name on top

# altering shape from linear list
# to list of two lists, each with two elements
CaseNames.shape = (nR,nC)

# each polar element requires a number
NumForPolar=1 # initial number for subplot name

for row in range(1,nR+1): 
#do this for each row
    for column in range(1,nC+1):
    #do this for each column (create  subplotplot)
        # get city name
        currentName=CaseNames[row-1,column-1]
        # create a name for the polar
        # this will be: 'polar1','polar2', etc
        polar_name='polar'+str(NumForPolar)
        # creating one subplot
        figINFO=go.Scatterpolar(
                # basic details for Scatterpolar
                r=safeRadarINS.loc[currentName],
                theta=VarNames,
                name=currentName,
                subplot=polar_name) #polar1,polar2, etc

        #adding a trace with previous info
        fig.add_trace(figINFO,
                      row, column)# location of subplot
        
        fig.update_layout({polar_name:layoutTrace,
                       'showlegend':False})
        NumForPolar+=1 # number for next polarName

 
fig.update_traces(fillcolor = 'silver',
                  line_color = 'silver',
                  fill='toself')
   
fig.update_annotations({'font':{'size':25,
                                'color':'black'},
                        "xanchor": "center",
                        "yanchor": "bottom",
                        "yref": "paper",
                        "borderpad":12})

# choose one:
## jupyter    
fig.show()
## spyder
plot(fig, auto_open=True)



#%%
