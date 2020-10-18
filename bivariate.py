# -*- coding: utf-8 -*-
"""
Python Codes for Chapter: 
Insights from TWO variables
"""

#%%

import pandas as pd

#link to data
linkRepo='https://github.com/resourcesbookvisual/data/'
linkFile='raw/master/crime.csv'
fullLink=linkRepo+linkFile
crime=pd.read_csv(fullLink)



#%%


#contingency table of counts
PrecintDaytime=pd.crosstab(crime.Precinct,crime.OccurredDayTime)

#marginal per column (column adds to 1)
PrecDayti_mgCol=pd.crosstab(crime.Precinct,crime.OccurredDayTime,
                            normalize='columns')

#making a data frame from contingency table
PrecDaytiDF=PrecintDaytime.stack().reset_index()
PrecDaytiDF.columns=["precint","daytime","counts"]

#adding marginal columns percents:
PrecDaytiDF['pctCol']=PrecDayti_mgCol.stack().reset_index().iloc[:,2]

# reformatting ordinal data
RightOrder=["day","afternoon","evening","night"]
PrecDaytiDF.daytime=pd.Categorical(PrecDaytiDF.daytime,
           categories=RightOrder,ordered=True)



#%%

# improved values for horizontal axis
minMargiPrecint=PrecintDaytime.apply(min,axis=1)
sortedPrecint=minMargiPrecint.sort_values(ascending=True)
newPrecintAxis=sortedPrecint.index

newPrecintAxis


#%%

from plotnine import *

base2= ggplot(PrecDaytiDF,
              aes(x='precint',y='counts',
                  fill='daytime')) + theme_classic()

barDodge2 = base2 +  geom_bar(stat="identity",
                              position="dodge",
                              color='grey')

barDodge2 += scale_fill_brewer(palette = "Greys") 

barDodge2 += scale_x_discrete(limits=newPrecintAxis)

barDodge2 += geom_text(aes(label='counts'),angle=0,
                       va='bottom',ha='center',
                       position=position_dodge(width=0.9))




barDodge2




#%%



# improved values for horizontal axis
minMargiPrecint=PrecintDaytime.apply(min,axis=1)
sortedPrecint2=minMargiPrecint.sort_values(ascending=False)
newPrecintAxis2=sortedPrecint2.index

newPrecintAxis2



#%%

# inversed copy
PrecDaytiDF['daytime2']=pd.Categorical(PrecDaytiDF.daytime,
                           categories=RightOrder[::-1],
                           ordered=True)

#%%
# ad-hoc set of colors         
adHoc=['white','white','black','black']

# manual colors according to Brewer.
from mizani.palettes import brewer_pal
Greys4=brewer_pal(palette='Greys')(4)[::-1]

#%%
base3= ggplot(PrecDaytiDF,
              aes(x='precint', y='counts',
                  fill='daytime2')) + theme_classic()
# order of horizontal
barStacked2c = base3 +scale_x_discrete(limits=newPrecintAxis2)
# manual Brewer palette
barStacked2c += scale_fill_manual(values=Greys4) 
# usual
barStacked2c += geom_bar(stat="identity", color='grey')
# annotating with color (default color will be assigned)
barStacked2c += geom_text(aes(label='counts',
                              color='daytime2'),
                         size = 8,
                         position = position_stack(vjust = 0.5))
# customized colors
barStacked2c += scale_colour_manual(values = adHoc) 
# use this to avoid text on top of legend symbols
barStacked2c += guides(color=False) # want to omit?

barStacked2c


#%%


from statsmodels.graphics.mosaicplot import mosaic

ax,t=mosaic(PrecintDaytime.stack(),gap=0.01)



#%%


base4= ggplot(PrecDaytiDF,
              aes(x='daytime',y='pctCol',
                  fill='precint')) + theme_classic()

barStPct2 = base4 + scale_fill_brewer(type='Qualitative',
                                      palette = "Paired") 

barStPct2 += theme(axis_title_y = element_blank(),
                   axis_text_y  = element_blank(),
                   axis_line_y  = element_blank(),
                   axis_ticks_major_y=element_blank())

barStPct2 += geom_bar(stat="identity",
                      position='fill')

barStPct2 += geom_text(aes(label='pctCol*100'),
                       format_string='{:.1f}%',
                       size = 8, fontweight='bold',
                       position = position_fill(vjust = 0.5))



barStPct2 



#%%



#contingency table of counts
CrimeDay=pd.crosstab(crime.crimecat,crime.OccurredDayTime)

#marginal per column (column adds to 1)
CrimeDay_mgCol=pd.crosstab(crime.crimecat,crime.OccurredDayTime,
                            normalize='columns')

#making a data frame from contingency table
CrimeDayDF=CrimeDay.stack().reset_index()
CrimeDayDF.columns=["crime","daytime","counts"]

#adding marginal columns percents:
CrimeDayDF['pctCol']=CrimeDay_mgCol.stack().reset_index().iloc[:,2]

# reformatting ordinal data
RightOrder=["day","afternoon","evening","night"]
CrimeDayDF.daytime=pd.Categorical(CrimeDayDF.daytime,
                                  categories=RightOrder,
                                  ordered=True)

# reformatting ordinal data
maxMargiCrime=CrimeDay.apply(max,axis=1)
sortedCrime=maxMargiCrime.sort_values(ascending=True)
newCrimeAxis=sortedCrime.index

newCrimeAxis



#%%


# reorder table vertically by max count per daytime
base5 = ggplot(CrimeDayDF, 
               aes(x='daytime',
                   y='crime')) + theme_minimal()

base5 += scale_y_discrete(limits=newCrimeAxis)

# plot value as point, size by value of percent
BTableDot = base5 + geom_point(aes(size = 'counts')) 

# label points, label with 2 decimal positions:
BTableDot += geom_text(aes(label = 'pctCol*100'),
                                    format_string='{:.2f}%',
                                    # push text to the right
                                    nudge_x = 0.3, 
                                    size=8)
# no need for legend
BTableDot += theme(legend_position="none") 



BTableDot


#%%


from mizani.formatters import percent_format

#crime ordered 
base6  = ggplot(CrimeDayDF, 
                 aes(x = 'crime',
                     y = 'pctCol') ) + theme_minimal()
#formatting text axis
base6 += scale_x_discrete(limits=newCrimeAxis)
base6 += scale_y_continuous(labels = percent_format())

#basic bar
BTableBar = base6 + geom_bar( stat = "identity" )

#Facetting
BTableBar += facet_grid('~ daytime') 

#Flipping
BTableBar += coord_flip() 

# altering axis text for crime
BTableBar += theme(axis_text_y = element_text(size=8,
                                              angle = 45,
                                              va='top')) 
# altering axis text for percent
BTableBar += theme(axis_text_x = element_text(size=6,
                                              angle = 45,
                                              va='bottom')) 

BTableBar


#%%

# heatplot

base7 = ggplot(CrimeDayDF,
               aes(x = 'daytime',y = 'crime',
                   fill = 'counts')) + theme_minimal()
base7 += scale_y_discrete(limits=newCrimeAxis)

# default heatplot
heat1 = base7 + geom_tile()
# customizing color
heat1 += scale_fill_gradient(low = "gainsboro",
                             high = "black")
# moving legend to the top
heat1 += theme(legend_title = element_blank(),
               legend_position="top")
# making legend colorbar wider
heat1 += guides(fill=guide_colorbar(barheight=200))




heat1 

#%% Preprocessing stage:



# counting crimes per year
crime.year.value_counts()

# keeping years since 2008
crime2=crime[crime.year>=2008].copy()

# making data without missing values
crime2.dropna(subset=['DaysToReport'],inplace=True)
crime2.fillna(value={'crimecat': 'UNcategorized'},inplace=True)

# sorting crimes for plotting
maxD=crime2.groupby('crimecat').describe()['DaysToReport'][['max']]
maxDSort=maxD.sort_values(by=['max'],ascending=True).index



#%%

#boxCrime_code

base8=ggplot(crime2,
             aes(x='crimecat',
                 y='DaysToReport')) + theme_minimal()
base8 += labs(x="crime") 
base8 += scale_x_discrete(limits=maxDSort)
boxCrime=base8 + geom_boxplot() + coord_flip()

boxCrime 

#%%
        
# computing quartiles 2 and 3
q23=['50%','75%'] # list of quartiles
crime2.groupby('crimecat').describe()['DaysToReport'][q23]



#%%
#subsetting
crimeYear=crime2[crime2.DaysToReport>365].copy()

#creating new variable
crimeYear['YearsToReport']=crimeYear.DaysToReport/365

# ordering the crimes by the q75 of YearsToReport
## subset with variables needed
subCrimeYear=crimeYear[['crimecat','YearsToReport']]
## grouping the subset by Q75
Q75=subCrimeYear.groupby('crimecat').quantile(q=0.75)
## sorting the grouping by Q75 of 'YearsToReport'
Q75Sort=Q75.sort_values(by=['YearsToReport'],ascending=True)
## just getting the names in order
sortedCrimesbyQ75=Q75Sort.index

sortedCrimesbyQ75
#%%

# plotting
base9=ggplot(crimeYear,
             aes(x='crimecat',
                 y='YearsToReport')) + theme_minimal()
base9 += labs(x="crime") 
#manually sorting
base9 += scale_x_discrete(limits=sortedCrimesbyQ75)
boxCrimeY = base9 + geom_boxplot() + coord_flip()

boxCrimeY 

#%%

# ad-hoc q3
import numpy as np

q3=lambda x:np.quantile(x,0.75)
theQ3='75%'## Labels for 'color'

# line of q3 grouped using last base9
q3Y = base9 + stat_summary(aes(group=True,color='theQ3'),
                               fun_y=q3,
                               geom='line', 
                               size=5)
# flipping
q3Y = q3Y + coord_flip()

q3Y 

#%%


#Labels for colors
theMin='Minima'
theMax='Maxima'

mq3Y = q3Y  + stat_summary(aes(group=True,color='theMin'),
                              fun_y=np.min,
                              geom='point', 
                              size=4)

Mmq3Y = mq3Y + stat_summary(aes(group=True,color='theMax'),
                             fun_y=np.max,
                             geom='point', 
                             size=1)

#customizing legend and colors
orderStats=["Minima","75%","Maxima"]
cols_orderStats=['silver','grey','black']
Mmq3Yfin= Mmq3Y + scale_color_manual(name='Stats',
                                     limits = orderStats,
                                     values = cols_orderStats) 

Mmq3Yfin


#%%


#formatting dates:
allCrimes = pd.value_counts(crime2.OccurredDate).reset_index()
allCrimes.columns=['dates','counts']
allCrimes['dates']=pd.to_datetime(allCrimes.dates, 
                                  format="%Y-%m-%d") 


#%%

baset= ggplot(allCrimes, 
            aes(x='dates', # already a DATE
                y='counts')) + theme_classic()
tsl = baset + geom_line(alpha=0.2) 


tsl


#%%

#pip install scikit-misc


# dots with some transparency (same 'baset')
tsp= baset + geom_point(alpha=0.2,
                         shape='+') 
#line for pattern
tsp += geom_smooth(fill='silver',
                    method='loess',
                    color='black')
#add format to axis:
tsp += scale_x_datetime(date_labels='%b-%Y',
                        date_breaks='6 months')
#set up text values on each tick:
tsp += theme(axis_text_x = element_text(angle=45))

tsp 




#%%


myArguments={'rule':'W', 'on':'dates', 'closed':'left'}
#using myArguments in resample with **
weekCrimes=allCrimes.resample(**myArguments).mean()

weekCrimes

#%%

basetW = ggplot(weekCrimes,
                aes(x='weekCrimes.index', #for index
                   y='counts')) + theme_classic()

tspW= basetW + geom_point(alpha=0.2,
                         shape='+') 

tspW += geom_smooth(fill='silver',
                    method='loess',
                    color='black')

tspW += scale_x_datetime(date_labels='%b-%Y',
                              date_breaks='6 months')

tspW += theme(axis_text_x = element_text(angle=45))

tspW 



#%%

# making table
crimeDate=pd.crosstab(crime2.OccurredDate,crime2.crimecat)
#table of dates to data frame
crimeDateDF=crimeDate.stack().reset_index()
#renaming columns
crimeDateDF.columns=['date','crime','count']
# formatting date column as Date type
crimeDateDF['date']=crimeDateDF['date'].astype('datetime64')

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

selection=["AGGRAVATED ASSAULT", 'WEAPON']
crimeDateDF_sub=crimeDateDF[crimeDateDF.crime.isin(selection)]

#extra step, get rid of values NOT used
crimeDateDF_sub.crime.cat.remove_unused_categories(inplace=True)

#%%

mini = pd.Timestamp("2014/1/1") #51
maxi = pd.Timestamp("2018/12/31")

basetSub=ggplot(crimeDateDF_sub,
                aes(x='date',y='count')) + theme_minimal()

tspSub  = basetSub + geom_point(alpha=0.3,
                                shape='x',
                                color='silver')

tspSub += geom_smooth(aes(color='crime'),
                      fill='white',size=2,
                      method='loess',alpha=1)

tspSub += scale_color_manual(values = ["grey", "black"])

tspSub += scale_x_datetime(date_labels='%b/%y',
                       date_breaks='2 months',
                       limits = [mini,maxi],
                       expand=[0,0]) 

tspSub += theme(legend_title = element_blank(),
                legend_position="top",
                axis_text_x = element_text(angle=90,
                                           va='top',
                                           size=6))

tspSub


#%%

baseTR  = ggplot(allCrimes,
                aes(x = 'counts')) + theme_void()
tsDens  = baseTR + geom_density(fill='grey')
tsRidge = tsDens + facet_grid("allCrimes.dates.dt.year~") 
tsRidge += theme(axis_text_x   = element_text())



tsRidge 

#%%


crime2015=crime[crime.year>=2015]
crime2015.dropna(inplace=True)



#%%
# operation to perform in each column:
operations={'DaysToReport': 'mean','Neighborhood': 'count'}
# grouping and applying operation
num_num=crime2015.groupby('Neighborhood').agg(operations)
# computing the total crimes
sumOfCrimes=num_num.Neighborhood.sum()
# overwriting column
num_num['Neighborhood']=100*num_num.Neighborhood/sumOfCrimes
# renaming data frame
num_num.columns=['meanDaysToReport','CrimeShare']
# Neighborhood is the index (row names), 
# moving it to a column
num_num.reset_index(inplace=True)


num_num




#%%
import scipy.stats  as stats

corVal,pVal=stats.spearmanr(num_num.meanDaysToReport,
                            num_num.CrimeShare)

corVal=str(round(corVal,2))
pVal=str(round(pVal,2))
TextCor='Spearman:\n' + corVal +'\n(p.value:'+ pVal +')'


#%%


baseNN = ggplot(num_num, 
                aes(x='meanDaysToReport',
                    y='CrimeShare')) + theme_minimal()
scat1  = baseNN +  geom_point(color='black')
#
scat2  = scat1 + geom_smooth(method = 'lm',
                          se=False,
                          color='silver')
scat2 += annotate(label=TextCor,
                       geom = 'text',
                       x=10,y=5)

scat2



#%%


#library needed
from statsmodels.formula.api import ols

#regression
relationship = ols('CrimeShare~meanDaysToReport',num_num).fit()
#influential values
influences = relationship.get_influence()
#saving Cook distance
num_num['cook'], pval = influences.cooks_distance
#computing thresold
threshold=4/len(num_num)
# condition
condition=np.where(num_num['cook'] > threshold,
                   num_num['Neighborhood'],"")


#%%

# needs installation: pip install adjustText
from adjustText import adjust_text

## parameters as a dictionary
# 'expand_objects'expand the bounding box of 
# texts when repelling them from other object
# 'arrowstyle' can also be '->' or '<-'
adjustParams= {'expand_points': (0,0), 
               'arrowprops': {'arrowstyle': '-',
                              'color': 'silver'}}


#%%

scat3 = scat1 + geom_text(aes(label=condition),
                          adjust_text=adjustParams) #repel!


scat3


#%%   
    
xVals=num_num.meanDaysToReport
yVals=num_num.CrimeShare
tVals=num_num.ConText


num_num['CondText']=np.where(xVals.between(3, 5) &
                            yVals.between(1, 3),
                            tVals,None)

#%%

import matplotlib.pyplot as plt

# size of plot
plt.figure(figsize=(10,6))

# plot hexabin
plt.hexbin(xVals,
           yVals,
           cmap=plt.cm.Greys, #colormap
           gridsize=10,
           mincnt=1) # at least

# color bar that represents counts
TheLegend = plt.colorbar()
TheLegend.ax.set_title('Legend\nTitle')

#%%



import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
# theme
plt.grid(color='silver') #lines color
plt.gca().set_facecolor("white") #background
#

plt.hexbin(xVals,
           yVals,
           cmap=plt.cm.Greys,
           gridsize=10,
           mincnt=1)
TheLegend = plt.colorbar()
TheLegend.ax.set_title('Legend\nTitle')


#%%


plt.figure(figsize=(10,6))
plt.grid(color='silver') 
plt.gca().set_facecolor("white") 
#
#ZOOMING IN
plt.axis([1.5, 7, 1, 3])
#
plt.hexbin(xVals,
           yVals,
           cmap=plt.cm.Greys,
           gridsize=10,
           mincnt=1)
#
#
# ANNOTATING
## settings of the label bounding box
for_bbox = {'boxstyle':"round",
              'fc':"white"}
## labels with bounding box
labels=[] #empty list of labels
for i in num_num.index:
    #creating each label
    labels.append(plt.text(xVals[i],yVals[i],
                           num_num.CondText[i],
                           color='gray',
                           fontsize=10,
                           bbox=for_bbox))
## repelling text    
adjust_text(labels,expand_text=(1.5, 1.5))
#
#
TheLegend = plt.colorbar()
TheLegend.ax.set_title('Legend\nTitle')

## titles 
plt.suptitle('The TITLE with matplotlib', y=1, fontsize=18)
plt.title('The SUBTITLE with matplotlib', y=1, fontsize=12)
plt.text(x=1.5,y=0.5,s="The Caption with matplotlib")
plt.xlabel('x-label with matplotlib')
plt.ylabel('y-label with matplotlib')



#%%

    
#%%    

# libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde

#LIMITS, THEME and ZOMING
plt.figure(figsize=(10,6))
plt.grid(color=None) 
plt.gca().set_facecolor("white") 
plt.axis([1.5, 7, 1, 3])
#
#
# MAKING THE DENSITY PLOT
## dividing plotting area
nbins=300
## computing densities
k = kde.gaussian_kde([xVals,yVals])
## preparing grid
xi, yi = np.mgrid[xVals.min():xVals.max():nbins*1j,
                  yVals.min():yVals.max():nbins*1j]
## preparing weights
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
## preparing color palette
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greys)
#
#
# ANNOTATING
for_bbox = {'boxstyle':"round",
              'fc':"white"}
labels=[] 
for i in num_num.index:
    labels.append(plt.text(xVals[i],yVals[i],
                           num_num.CondText[i],
                           color='gray',
                           fontsize=10,
                           bbox=for_bbox))   
adjust_text(labels,expand_text=(1.5, 1.5))
#
#
#TITLES
plt.suptitle('The TITLE with matplotlib', fontsize=18)
plt.title('The TITLE with matplotlib', fontsize=12)
plt.text(x=1.5,y=0.75,s="The Caption with matplotlib")
plt.xlabel('x-label')
plt.ylabel('y-label')
plt.grid(color='silver')
plt.show()





#%%




#%%
crimeDateDF_sub.index=crimeDateDF_sub['date']

countDF=crimeDateDF_sub.groupby([pd.Grouper(freq='W'), 'crime']).sum().reset_index()
countDFWide=countDF.pivot(index='date', columns='crime', values='count')
countDFWidePCT=countDFWide.pct_change().dropna()

#%%

countDFLongPCT=countDFWidePCT.stack().reset_index()

countDFLongPCT.columns=['date','crime','change']

#%%

ggplot(countDFLongPCT[abs(countDFLongPCT.change)>0.75],
              aes(x='date',
                  y='change')) + geom_point(color='silver') + geom_smooth(aes(color='crime'))#facet_wrap('crime',ncol=1)



#%%

import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np


grangercausalitytests(countDFWidePCT,2)