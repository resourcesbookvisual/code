# -*- coding: utf-8 -*-
"""
Python Codes for Chapter: 
Insights from ONE variable
"""

#%%

import pandas as pd

#link to data
linkRepo='https://github.com/resourcesbookvisual/data/'
linkFile='raw/master/eduwa.csv'
fullLink=linkRepo+linkFile
eduwa=pd.read_csv(fullLink)
#
# Frequency table
FTloc = pd.value_counts(eduwa.LocaleType,
                     ascending=True,
                     dropna=False).reset_index()
FTloc.columns = ['Location','Count']
# adding column 
FTloc['Percent']=100*(FTloc.Count/FTloc.Count.sum()).round(4)
# 
FTloc['Location'].fillna('Uncategorized', inplace=True)
# new column with gap value
FTloc['Gap']=(FTloc.Percent-25).round(0)
# new column with True if gap is positive (False otherwise)
FTloc['Above_Equal_Share']=FTloc.Gap>0



#%%

texts={'TITLE':"Distance from even distribution",
       'sTITLE':"Location of Schools in WA State (2018)",
       'SOURCE':"Source: US Department of Education"}

rePOSITION=FTloc.Location

from plotnine import *

info1=ggplot(FTloc,aes(x='Location',y='Gap',label='Gap'))
barFT1= info1 + geom_bar(stat = 'identity',width=0.4)
# theme 
barFT1= barFT1 + theme_classic() 
#erasing
barFT1= barFT1 + theme(axis_ticks= element_blank(),
                       axis_text_y = element_blank(),
                       axis_title_y = element_blank(),
                       axis_title_x = element_blank(),
                       axis_line_y = element_blank())
#repositioning
barFT1= barFT1 + scale_x_discrete(limits=rePOSITION)
#threshold
barFT1=barFT1 + geom_hline(aes(yintercept=0)) 
#text in bars
barFT1=barFT1 + geom_label(aes(label=rePOSITION),
                           y=0,size=9,fill='silver') + geom_label()

# text with matplotlib
import matplotlib.pyplot as plt

fig = barFT1.draw()
fig.text(x=0.1,y=-0.02,s=texts['SOURCE'])
plt.suptitle(texts['TITLE'], y=0.97, fontsize=14)
plt.title(texts['sTITLE'], fontsize=10)
#plt.show()

 


  
#
# 
# 
# 
#
#



#%%

# changes in aes: 
info2 = ggplot(FTloc, aes(x='Location',
                        y='Gap', 
                        color='Above_Equal_Share',
                        label='Gap'))
# one for the lollipop stick
lol1=info2 + geom_segment(aes(x = 'Location',xend = 'Location',
                              y = 0,yend = 'Gap'),#here
                          color = "lightgray")
# one for the lollipop head (just a dot)
lol1 = lol1 + geom_point(size=10) 
# NO CHANGES (no realigment needed in Plotine)
lol1 = lol1 + theme_classic()
# NO X-AXIS
lol1 = lol1 + theme(axis_ticks= element_blank(),
                    axis_text_y = element_blank(),
                    axis_title_y = element_blank(),
                    axis_line_y = element_blank(),
                    axis_text_x = element_blank(),
                    axis_line_x = element_blank(),
                    axis_title_x = element_blank())
# NO CHANGES (title later)
lol1 = lol1 + scale_x_discrete(limits=rePOSITION)
# annotating threshold
lol1 = lol1 + geom_hline(yintercept=0, linetype = "dashed")
# for 'Gap' values
lol1 = lol1 + geom_text(show_legend = False,color='white',size=8)
# for 'Location' values.
lol1 = lol1 + geom_label(aes(label=rePOSITION),
                         color='black',size=9,
                         y=0, show_legend = False) 
# coloring
lol1 = lol1 + scale_color_gray(0.6,0.2)
# legend position and frame
lol1 = lol1 + theme(legend_position = (0.8,0.4), 
                    legend_background = element_rect(linetype="solid", 
                                                     colour ="grey"))
# text with matplotlib
import matplotlib.pyplot as plt

fig = lol1.draw()
fig.text(x=0.1,y=-0.02,s=texts['SOURCE'])
plt.suptitle(texts['TITLE'], y=0.97, fontsize=14)
plt.title(texts['sTITLE'], fontsize=10)




#%%
#link to data
linkRepo='https://github.com/resourcesbookvisual/data/'
linkREPS='raw/master/reps.csv'
fullLink=linkRepo+linkREPS
reps=pd.read_csv(fullLink)
# Frequency table
FTrep = pd.value_counts(reps.Residence,
                     ascending=True,
                     dropna=False).reset_index()
FTrep.columns = ['Residence','Legislators']


#%%

# text for titles
texts_2={'TITLE':"Cities represented by Legislators",
         'sTITLE':"WA State House of Representative (2019-2021)",
         'SOURCE':"Source: Washington State Legislature."}
# text for tick labels
rePOSITION_2=FTrep.Residence
CountToShow=FTrep.Legislators
#base
info4 = ggplot(FTrep, aes(x='Residence',
                          y='Legislators'))
# Lollipop stick
lol2= info4 + geom_segment(aes(y = 0,
                               yend = 'Legislators',
                               x = 'Residence',
                               xend = 'Residence'),
                          color = "black")
# Lollipop head
lol2= lol2 + geom_point(size=2) 
# theme: 
lol2 = lol2 + theme_classic()
# titles and aligment later in matplotlib
# repositioning
lol2 = lol2 + scale_x_discrete(limits=rePOSITION_2)
#flipping
lol2 = lol2 + coord_flip()
# Vertical axis changes
lol2 = lol2 + theme(axis_title_y=element_blank(),
                    axis_text_y=element_blank(),
                    axis_ticks_major_y = element_blank(),
                    axis_line_y = element_blank())
# Horizontal axis changes
lol2 = lol2 + scale_y_continuous(breaks=CountToShow)
lol2 = lol2 + theme(axis_line_x = element_blank())
lol2 = lol2 + theme(panel_grid_major_x = 
                    element_line(color = "silver",
                                 linetype = "dashed"))
# annotations: text near dot
lol2 = lol2 + geom_text(aes(label='Residence'),
                        ha = 'left',
                        nudge_y = 0.1,
                        size=4.5)

#%%

# text with matplotlib
import matplotlib.pyplot as plt

fig = lol2.draw()
fig.text(x=0.1,y=-0.02,s=texts_2['SOURCE'])
plt.suptitle(texts_2['TITLE'], y=0.97, fontsize=14)
plt.title(texts_2['sTITLE'], fontsize=10)

#%%

import pandas as pd

#link to data
linkRepo='https://github.com/resourcesbookvisual/data/'
linkFile='raw/master/crime.csv'
fullLink=linkRepo+linkFile
crime=pd.read_csv(fullLink)

# Frequency table
FTcri = pd.value_counts(crime.cat,
                     ascending=False,
                     dropna=False).reset_index()
FTcri.columns = ['Crimes','Counts']
# adding column 
FTcri['CumPercent']=100*FTcri.Counts.cumsum()/FTcri.Counts.sum()
# renaming missing values
FTcri['Crimes'].fillna('UNCATEGORIZED', inplace=True)


#%%

import numpy as np

condition=FTcri.Crimes.isin(FTcri.Crimes[0:4])
TOPS =tuple(np.where(condition, 'black', 'silver'))
 

#%%


from plotnine import *

#1
info5=ggplot(FTcri,aes(x='Crimes',y='CumPercent')) + theme_classic() 
#2
annot1=info5 + geom_hline(yintercept = 80, linetype='dashed') 
#3
cumBar1=annot1 + geom_bar(stat = 'identity',
                           fill='white',
                           color=TOPS,
                           width = 0.2)
#4
cumBar1=cumBar1+ scale_x_discrete(limits=FTcri.Crimes)
#5
cumBar1=cumBar1+ scale_y_continuous(breaks = (20,50,80,100)) 
#6a
cumBar1=cumBar1+ theme(axis_text_x=element_text(rotation=45,
                                                ha='right'))




#%%

# 6b
import matplotlib.pyplot as plt

fig = cumBar1.draw()

ax=plt.subplot() # the plot area
# coloring each label
for aTick, aColor in zip(ax.get_xticklabels(), TOPS):
    aTick.set_color(aColor)




#%%



from paretochart import pareto

#using pareto
ax=pareto(FTcri.Counts, FTcri.Crimes,
       line_args=('k'), #'k' is 'black'
       data_kw={'color': 'grey'}) 

#reference lines using matplotlib
plt.axvline(x=3,ls='--',c='silver',lw=1)
plt.axhline(y=80,ls='--',c='silver',lw=1)

#modifying tick labels
plt.setp(ax[1].get_xticklabels(), ha="right",rotation=45)
plt.show()


#%%


from pandas.api.types import CategoricalDtype

# get the levels
levels=eduwa['High.Grade'].value_counts().index.sort_values()
#reorder levels
ordLabels=levels[-2:].tolist()[::-1]+sorted(map(int,levels[:-2]))
# turn previous result into a list of strings
ordLabels=list(map(str,ordLabels))
# use that list of levels to create ordinal format
HGtype= CategoricalDtype(categories=ordLabels,ordered=True)
# apply that format to the column
eduwa['High.Grade']= eduwa['High.Grade'].astype(HGtype)



#%%


# Frequency table
FThg = pd.value_counts(eduwa['High.Grade'],
                        ascending=False,sort=False,
                        dropna=False).reset_index()
FThg.columns = ['MaxOffer','Counts']
# adding column 
FThg['CumPercent']=100*FThg.Counts.cumsum()/FThg.Counts.sum()

#%%


# function for quartiles in ordinal data
def Quart_Pos(cumFT,q=1): # q can be 1,2 or 3.
    position=0
    for percent in cumFT:
        if percent <= 25*q: position +=1 
        else: break
    return position # returns a position

# applying function 
medianHG=FThg.MaxOffer[Quart_Pos(FThg.CumPercent,2)]


#%%

import numpy as np

# color to highlight median
condition=[medianHG==test for test in ordLabels]
colCondition=tuple(np.where(condition, 'black', 'silver'))

# usual
info7=ggplot(FThg, aes('MaxOffer','Counts')) + theme_classic()
barFThg=info7 + geom_bar(stat='identity',fill=colCondition)
barFThg=barFThg + scale_x_discrete(limits=ordLabels)

#barFThg



#%%
# from ordinal to numeric
eduwa['High.Grade.Num']=eduwa['High.Grade'].cat.codes

info8=ggplot(eduwa,aes(x=0,y='High.Grade.Num')) + theme_classic()
boxHG =info8+ geom_boxplot() 
boxHG =boxHG + coord_flip()
boxHG = boxHG + scale_y_continuous(labels=ordLabels,
                                   breaks=list(range(0, 15)))

boxHG





#%%

theBreaks=list(range(0, 15))

info8=ggplot(eduwa,aes(x=0,y='High.Grade.Num')) + theme_classic()
vio1 =info8 + geom_violin(width=1.4,
                          fill="black", color=None) 
boxHG2 = vio1 + geom_boxplot(width=0.2,
                             fill='white',
                             color='silver',
                             fatten=4) 
boxHG2 = boxHG2 + coord_flip()
boxHG2 = boxHG2 + scale_y_continuous(labels=ordLabels,
                                     breaks=theBreaks)
boxHG2 = boxHG2 + theme(axis_ticks = element_blank(),
                      axis_text_y = element_blank(),
                      axis_title_y = element_blank(),                      
                      axis_line_y  = element_blank())

boxHG2




#%%

# get stats as DICT: count, mean,std,min,q1,q2,q3,max  
statVals=eduwa['Reduced.Lunch'].describe().to_dict()

# Turn into DICT of integers (rounding up)
from math import ceil
statVals={key: ceil(val) for key, val in statVals.items()}

# Thresholds to detect outliers:

## distance between quartiles
IQR=statVals['75%']-statVals['25%']
## Thresholds:
statVals['upper']=1.5*IQR + statVals['75%']
statVals['lower']=statVals['25%'] - 1.5*IQR

#prepare annotations:
axisKeys=['min','25%','50%','mean','75%','upper','max']
myTicks = {axKey: statVals[axKey] for axKey in axisKeys}

# Share of values considered outliers:
theVariable=eduwa['Reduced.Lunch']
theVariable = theVariable.dropna()
countOutliersUp=sum(theVariable>statVals['upper'])
shareOut=ceil(countOutliersUp/len(theVariable)*100)
# message using the value computed:
labelOutliers="Outliers:\n" + str(shareOut) + "% of data"




#%%




info10= ggplot(eduwa,aes(x=0,y = 'Reduced.Lunch')) + theme_classic()
info10=info10 + scale_y_continuous(breaks =list(myTicks.values())) 
info10=info10 +xlim(-0.25,0.3) +coord_flip()
disp2= info10 + geom_boxplot(width=0.25,outlier_alpha = 0.2)

# annotating
## standard deviation
disp2=disp2 + annotate("pointrange", 
                       x=0.15, y=statVals['mean'],
                       ymin = (statVals['mean']+5)-statVals['std'], 
                       ymax = (statVals['mean']+5)+statVals['std'],
                       colour = "silver", size = 1)
## mean
disp2=disp2 + geom_hline(yintercept = statVals['mean'],
                         linetype='dotted')
disp2= disp2 + annotate(geom="text",
                        x=0.2, y=statVals['mean']+5, 
                        label="Mean",angle=90,size=10)
## median
disp2=disp2 + geom_hline(yintercept = statVals['50%'],
                         linetype='dotted')
disp2= disp2 + annotate(geom="text", 
                        x=-0.2, y=statVals['50%']-5, 
                        label="Median",angle=90,size=10)

## outliers
disp2=disp2 + geom_hline(yintercept = statVals['upper'],
                         linetype='dashed',color='silver')
disp2=disp2 + annotate(geom="label", 
                       x=0.1,y=statVals['max'],
                       label=labelOutliers,size=12,ha='right',
                       color='silver')
# erasing
disp2=disp2 + theme(axis_ticks_major_y = element_blank(),
                    axis_line_y = element_blank(),
                    axis_text_y = element_blank(),
                    axis_title_y = element_blank())


disp2

#%%

#1
theStart=statVals['min']
width=10
#2
oldMax=statVals['max']
reminderMax=oldMax%width
newMax= oldMax+(width-reminderMax) if reminderMax<width else oldMax
#3
TheBreaks=list(range(theStart,newMax+width,width))
#4
intervals=pd.cut(eduwa['Reduced.Lunch'],
              bins=TheBreaks,include_lowest=True)
topCount=intervals.value_counts().max()
#5
widthY=50
reminderY=topCount%widthY
top_Y=topCount+widthY-reminderY if reminderY<widthY else topCount
vertiVals=list(range(0,top_Y+widthY,widthY))
#%%

#1
N = statVals['count']
MEAN = statVals['mean']
STD = statVals['std']
#2
def NormalHist(x,m=MEAN,s=STD,n=N,w=width): 
    import scipy.stats as stats
    return stats.norm.pdf(x, m, s)*n*w


#%%


info11= ggplot(eduwa, aes(x = 'Reduced.Lunch')) + theme_classic()
disp3= info11 + geom_histogram(binwidth = width,
                               boundary=theStart,
                               fill='white',color='silver') 
disp3= disp3 + stat_function(fun = NormalHist,
                  color = "black", size = 1,linetype='dashed')

disp3= disp3 + scale_y_continuous(breaks = vertiVals)


disp3




#%%

import ineqpy as ineq

#new data frame (one column)
dfTest=pd.DataFrame(eduwa['Reduced.Lunch'].dropna())
## add a columns, where each value is number '1':
dfTest['count']=np.ones(dfTest.size) 


# data frame to survey object
dfIneq = ineq.api.Survey(dfTest, weights='count')



#for titles
txtLz={'HorizontalTitle':"Percent of Schools by benefit received",
       'VerticalTitle':"Cummulative percent of benefit",
       'plotTitle':"How much receives\nthe 20% that receives most?",
       'sourceText':"Source: US Department of Education"}

# text for annotation
## computing
gini=dfIneq.gini('Reduced.Lunch')
## pasting message (number to text before pasting)
GINItext='Gini:' + str(gini.round(3))

# plot diagonal (automatic) and Lorenz curve (in that order)
symbols=['k--',"k-"] # color (black) and line type (respect order).
ax1=dfIneq.lorenz('Reduced.Lunch').plot(legend=False,style=symbols)

# annotations
## vertical and horizontal lines (matplotlib)
plt.axvline(x=0.8,ls=':',c='silver',lw=1)
plt.axhline(y=0.5,ls=':',c='silver',lw=1)

# changing default axis tick values, positions and aspect
plt.yticks((0,0.5,0.8))
plt.xticks((0,0.5,0.8))
ax1.yaxis.set_label_position("right")
ax1.yaxis.set_ticks_position("right")
plt.axes().set_aspect('equal')

# annotating: adding Gini Index value 
plt.text(0.4, 0.25,GINItext, fontsize=8)

# texts
plt.title(txtLz['plotTitle'])
plt.ylabel(txtLz['VerticalTitle'])
plt.xlabel(txtLz['HorizontalTitle'])


#plt.show()



#%%







