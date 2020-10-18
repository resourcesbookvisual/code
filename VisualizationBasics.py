# -*- coding: utf-8 -*-
"""
Python Codes for Chapter: 
visualizatio nBasics
"""




#%%
import pickle
from urllib.request import urlopen
linkRepo='https://github.com/resourcesbookvisual/data/'
linkDemo="raw/master/demo.pkl" # 'pickle' file!

demo = pickle.load(urlopen(linkRepo+linkDemo))


#%%

from plotnine import *

info=ggplot(demo, aes(x='Culture', y='Functioning'))
dots1=info + geom_point(shape="*", size=2)
lines1=dots1 + geom_smooth(method = "lm",se=False,colour="black")
lines1 + theme_classic()


#%%


lines2=dots1 + geom_smooth(se=False,colour="black")
lines2 + theme_classic()


#%%

# https://matplotlib.org/3.1.1/api/markers_api.html
info=ggplot(demo,aes(x='Culture',y='Functioning',size='Regime'))
polyg1 = info + geom_point(shape ="x")
polyg1 + theme_classic()



#%%


info=ggplot(demo,aes(x='Culture',y='Functioning',
                     colour='Continent'))
colorNom1=info+geom_point(size=3) 
colorNom1 + scale_colour_brewer(type='qual',palette = "Set1") + \
            theme_classic()
            
            
#%%
            
            
info2=ggplot(demo,aes(x='Culture',y='Functioning',fill='Regime'))
colorOrd3 = info2 + geom_point(size=3,shape='o')
colorOrd3 + scale_fill_brewer(palette = "OrRd") + theme_classic()



#%%


info4=ggplot(demo,aes(x='Culture',y='Functioning',
                      fill='Electoral'))
colorNum=info4 + geom_point(size=3, shape='o')
colorNum + scale_fill_gradient2(midpoint = 5,
                                mid= 'white',
                                low = '#e66101',
                                high = '#5e3c99')



#%%

import pandas as pd

FT = pd.value_counts(demo.Continent,ascending=True).reset_index()
FT.columns = ['Values','Counts']



#%%


# Titles to be used:
the_Title="A NICE TITLE"
the_SubTitle="A nice subtitle"
TheTopTitles=the_Title+'\n'+the_SubTitle  # adaptation
horizontalTitle="Continents present in the study"
verticalTitle="Number of countries studied"
# data for annotation
theCoordinates={'X':1,'Y':7} #dict instead of list
theMessage="So few?!"



#%%



info=ggplot(FT, aes(x='Values',y='Counts'))
titles1= info + ggtitle(TheTopTitles)
titles2= titles1 + xlab(horizontalTitle) + ylab(verticalTitle)

annot1= titles2 + annotate("text", 
                      x = theCoordinates['X'], #reading dict
                      y = theCoordinates['Y'], 
                      label = theMessage)
align1= annot1 + theme(plot_title = element_text(ha = "center"),
                        axis_title_y = element_text(ha = "top"))

barFT1= align1 + geom_bar(stat = 'identity') 
barFT2= barFT1 + theme_classic()
barFT3= barFT2 + theme(axis_title_x = element_blank(),
                       axis_title_y = element_text(ha = "center"),
                       axis_ticks_major_x = element_blank(),
                       axis_ticks_major_y = element_blank(),
                       axis_text_x = element_text(va = 'top',
                                                  size=6),
                       plot_title = element_text(ha = "center"),
                       axis_line_x = element_blank())

barFT4 = barFT3 + scale_x_discrete(limits=FT.Values)

#%%

barFT4 + geom_text(aes(label='Counts'), 
                   va='top', 
                   color="white", size=8) + \
                   theme(axis_text_y = element_blank())
                   
                   
                   
#%%
                   
                   
barFT4 + theme(panel_grid_major_y=element_line(color="grey")) +\
         scale_y_continuous(breaks=FT.Counts)
   
      
#%%

info=ggplot(demo, aes(x='Culture', y='Functioning',
                      shape='Continent'))
leyenda=info + geom_point()
erase2=leyenda + theme_classic() + \
                 theme(legend_title=element_blank())
repo1=erase2 + theme(legend_position="top")
repo2=repo1 + coord_fixed(ratio=1) + \
              guides(shape=guide_legend(nrow=1))
repo2 + theme(legend_key_width = 0,
              legend_background = element_rect(size=0.5,
                                       linetype="solid",
                                       colour ="grey")) 


#%%

## plotnine

info=ggplot(demo, aes(x='Culture', y='Functioning',
                      shape='Continent'))
leyenda=info + geom_point()
erase2=leyenda + theme_classic() + \
                 theme(legend_title=element_blank())
repo2=erase2 + coord_fixed(ratio=1) 
repo2=repo2 + theme(legend_key_width = 0,
              legend_background = element_rect(size=0.5,
                                       linetype="solid",
                                       colour ="grey")) 

# HELP from matplotlib
import matplotlib.pyplot as plt
fig = repo2.draw()
# x=0,y=0 is lower left corner; x=1,y=1 is upper right
fig.text(x=0.7,y=0.01,s="The Caption with matplotlib")
# suptitle for TITLE
plt.suptitle(the_Title, y=0.95, fontsize=18)
# title for subTITLE
plt.title(the_SubTitle, fontsize=10)