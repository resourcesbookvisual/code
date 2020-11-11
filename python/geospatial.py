#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Codes for Chapter: 
Insights from MAPs

@author: JoseManuel
"""
#%%
import pandas as pd

link1='https://github.com/resourcesbookvisual/data/'
link2='raw/master/contriWA.csv'
contriWA=pd.read_csv(link1+link2)
 
 

#%%
import geopandas as gpd

myGit="https://github.com/resourcesbookvisual/data/"
myGeo="raw/master/WAzipsGeo.json"
mapLink=myGit + myGeo

wazipMap = gpd.read_file(mapLink)

#%%

wazipMap.plot()

#%%
import geoplot as gplt
gplt.polyplot(wazipMap)


#%%

wazipMap.ZCTA5CE10.describe()

#%%

# Filtering
#conditions
condition1='election_year==2012 and cash_or_in_kind=="Cash"'
condition2='party.isin(["DEMOCRAT","REPUBLICAN"])'

#chained queries
contriWASub=contriWA.query(condition1).query(condition2)
#%%
# Aggregating
contrisum=contriWASub.groupby(['contributor_zip','party'])
contrisum=contrisum.agg({'amount':'sum'}).reset_index().fillna(0)
contrisum=contrisum.pivot(index='contributor_zip', 
                          columns='party', 
                          values='amount').reset_index().fillna(0)
contrisum['total']=contrisum.DEMOCRAT + contrisum.REPUBLICAN


contrisum
#


#%%%

wazipMap=wazipMap.loc[:,['ZCTA5CE10','geometry']]
# contributor_zip as text (it was a number)
contrisum.contributor_zip=contrisum.contributor_zip.astype(str)
allZip=wazipMap.merge(contrisum,
                      left_on='ZCTA5CE10',
                      right_on='contributor_zip',
                      how='left')

#%%
import numpy as np

comparison=allZip.REPUBLICAN>allZip.DEMOCRAT
condition=allZip.loc[:,["REPUBLICAN","DEMOCRAT"]].any(axis=1)
allZip['winnerREP']=condition1
allZip['winnerREP']=np.where(condition,
                             comparison,
                             None)


#%%

allZip.plot(column='winnerREP', #column to color
            categorical=True,
            edgecolor='black',
            legend=True,
            missing_kwds={'color': 'lightgrey'},
            cmap='gist_gray') # palette for column chosen



#%%

#missing
allZip[allZip.winnerREP.isnull()].shape[0]


#%%


# dissolve
## a. make copy
waBorder=wazipMap.copy()
## b. Correct map with buffer (may not be needed)
waBorder['geometry'] = waBorder.buffer(0.01)
## c. create a constant column by which to dissolve
waBorder['dummy']=1
## d. dissolving
waBorder= waBorder.dissolve(by='dummy')
## e. plot the dissolved map
waBorder.plot(color='white',edgecolor='black')


#%%


# dissolve
## a. make copy
allZipBetter=allZip.copy()
## a.1 saving missing values
NAs = allZipBetter[allZipBetter.winnerREP.isna()]
## b. Correct map with buffer (may not be needed)
allZipBetter['geometry'] = allZipBetter.buffer(0.01)
## c. dissolving
allZipREP= allZipBetter.dissolve(by='winnerREP',as_index=False)
## d. plotting the dissolved map
allZipREP.append(NAs).plot(column='winnerREP', #column to color
               edgecolor='black',
               categorical=True,legend=True,
               missing_kwds={'color': 'grey'}, 
               cmap='gist_gray') 

#%%


# turn lon lat into geoDataFrame 
WApoints = gpd.GeoDataFrame(contriWASub, 
                geometry=gpd.points_from_xy(contriWASub.Lon,
                                            contriWASub.Lat))
WApoints.crs = wazipMap.crs

WApoints.plot()

#%%

## a. make polygons with missing values
allZipNA=allZip[allZip.total.isnull()].copy()
## b. Correct map with buffer (may not be needed)
#allZipNA['geometry'] = allZipNA.buffer(0.01)
## c. create a constant column by which to dissolve
#allZipNA['dummy']=1
## d. dissolving
#allZipNA= allZipNA.dissolve(by='dummy',as_index=False)


#%%

#plot with default projection in geopandas
layerBorder= waBorder.plot(edgecolor='grey',color='white')

layerMissing=allZipNA.plot(edgecolor='grey',color='grey',
                           ax=layerBorder)
WApoints.plot(color='black', 
              markersize=0.1,alpha=0.1,
              ax=layerBorder) # on top of!)






#%%

#plot with default projection in geoplot
layerBorder = gplt.polyplot(waBorder,
                            edgecolor='grey',
                            facecolor='white')
layerMissing = gplt.polyplot(allZipNA,
                            edgecolor='grey',
                            facecolor='grey',
                            ax=layerBorder)
gplt.pointplot(WApoints,
               color='black',
               s=0.1,#size of point
               alpha=0.1,
               ax=layerBorder)# on top of!


#%%

#reprojecting with geopandas
layerBorder= waBorder.to_crs("EPSG:3395").plot(edgecolor='grey', 
                                               color='white')

layerMissing=allZipNA.to_crs("EPSG:3395").plot(edgecolor='grey',
                                               color='grey',
                                               ax=layerBorder)

WApoints.to_crs("EPSG:3395").plot(color='black',
                                  markersize=0.1,
                                  alpha=0.1,
                                  ax=layerBorder)


#%%

import geoplot.crs as gcrs #activating!

layerBorder = gplt.polyplot(waBorder, 
                            projection=gcrs.Mercator(),#HERE!
                            edgecolor='grey', 
                            facecolor='white')

layerMissing = gplt.polyplot(allZipNA,
                             edgecolor='grey',
                             facecolor='grey',
                             ax=layerBorder)
layerPoint= gplt.pointplot(WApoints,
                           color='black',
                           s=0.1,
                           alpha=0.1,
                           ax=layerBorder)# on top of!


#%%

myGit="https://github.com/resourcesbookvisual/data/"
myGeo2="raw/master/waCountiesfromR.geojson"
mapLink2=myGit + myGeo2

waCounties= gpd.read_file(mapLink2)

kingMap=waCounties[waCounties.JURISDIC_2=="King"]

#%%

import matplotlib.pyplot as plt

fig, ax = plt.subplots()#plt.subplots(figsize=(10,6))
# zooming area:
ax.set_xlim(kingMap.total_bounds[0:4:2]) #recovering indices
ax.set_ylim(kingMap.total_bounds[1:5:2]) #recovering indices
# maps of WASHINGTON
Border= waBorder.plot(edgecolor='grey',color='white',ax=ax)
Zips=wazipMap.plot(edgecolor='silver',color='whitesmoke',ax=Border)
# ZOOMING IN:
WApoints.plot(color='black',markersize=0.1,alpha=0.1,ax=Border)
plt.show()




#%%
# keeping non-missing data
allZip=allZip[~allZip.total.isnull()]

#plotting absolute values
#forDems
Border= waBorder.plot(edgecolor='grey',color='red')
allZip.plot(column='DEMOCRAT',ax=Border)

#forAll
Border= waBorder.plot(edgecolor='grey',color='red')
allZip.plot(column='total',ax=Border)

#plotting relative values
Border= waBorder.plot(edgecolor='grey',color='red')
allZip['DemChoro'] = allZip.DEMOCRAT / allZip.total
allZip.plot(column='DemChoro',legend=True,ax=Border,
            legend_kwds={'shrink': 0.6}) #shrink legend size

#%%

## geoplot
#plotting absolute values
#forDems
Border = gplt.polyplot(waBorder,
                       edgecolor='grey', 
                       facecolor='red')
gplt.choropleth(allZip, hue='DEMOCRAT',ax=Border)

#forAll
Border = gplt.polyplot(waBorder,
                       edgecolor='grey', 
                       facecolor='red')
gplt.choropleth(allZip, hue='total',ax=Border)

#plotting relative values
Border = gplt.polyplot(waBorder,
                       edgecolor='grey', 
                       facecolor='red')
gplt.choropleth(allZip, hue='DemChoro',ax=Border, legend=True)



#%%


link3='raw/master/covidCountyWA.csv'
LINK=link1 + link3
#getting the data TABLE from the file in the cloud:
covid=pd.read_csv(LINK)

#%%

covidMap=pd.merge(waCounties.loc[:,["JURISDIC_2","geometry"]],covid,
               left_on="JURISDIC_2",right_on="County")


#%%

import mapclassify as mc

#well-known styles
#geopandas for Equal intervals
covidMap.plot(column='CasesPer100k',
              scheme='EqualInterval', 
              k=5,
              cmap='OrRd',
              legend=True)
#%%
#geopandas for Quantiles
covidMap.plot(column='CasesPer100k',
              scheme='Quantiles', 
              k=5,
              cmap='OrRd',
              legend=True)

#%%

#well-known styles
#geoplot for Equal intervals
gplt.choropleth(covidMap, 
                hue='CasesPer100k', 
                scheme=mc.EqualInterval(covidMap.CasesPer100k,k=5),
                cmap='OrRd',
                legend=True)
#%%
#geoplot for Quantiles
gplt.choropleth(covidMap, 
                hue='CasesPer100k', 
                scheme=mc.Quantiles(covidMap.CasesPer100k, k=5),
                cmap='OrRd',
                legend=True)

#%%
#optimization styles in geopandas
covidMap.plot(column='CasesPer100k',
              scheme='FisherJenks', 
              k=5,
              cmap='OrRd',
              legend=True)
#%%
covidMap.plot(column='CasesPer100k',
              scheme='HeadTailBreaks',
              cmap='OrRd',
              legend=True)
#%%
#optimization styles in geoplot
gplt.choropleth(covidMap, 
                hue='CasesPer100k', 
                scheme=mc.FisherJenks(covidMap.CasesPer100k, k=5),
                cmap='OrRd',
                legend=True)
#%%
gplt.choropleth(covidMap, 
                hue='CasesPer100k', 
                scheme=mc.HeadTailBreaks(covidMap.CasesPer100k),
                cmap='OrRd',
                legend=True)

#%%
#new variable
valuesNew=100000*(covidMap.Deaths/covidMap.Population)
covidMap['DeathsPer100k']=valuesNew

#%%
import geoplot.crs as gcrs #activating!

#border
border=gplt.polyplot(df=covidMap,
                     projection=gcrs.Mercator(),#reproject
                     edgecolor='gray', #border
                     facecolor='gainsboro') #fill of polygon
#area and color
gplt.cartogram(df=covidMap,#map
               scheme=mc.EqualInterval(covidMap.DeathsPer100k,k=3),
               cmap=plt.get_cmap('YlGnBu',3),#palette
               hue="DeathsPer100k", #var for color                   
               scale='CasesPer100k',#var for resize
               limits=(0.3, 1),  #limits cartogram polygons       
               edgecolor='None', #no border
               legend=True,
               legend_var='hue', #legend of what
               legend_kwargs={'bbox_to_anchor': (0.1, 0.4),#location
                              'frameon': True, #with frame?
                              'markeredgecolor':'k',
                              'title':"DeathsPer100k"},
               ax=border)
#%%
# mapOfPoints

import geoplot.crs as gcrs #activating!

#borders
countyBorders = gplt.polyplot(df=covidMap, 
                              projection=gcrs.Mercator(),#HERE!
                              edgecolor='grey',
                              facecolor='gainsboro')

#points scaled
covidMap['centroid']=covidMap.centroid #compute centroids
gplt.pointplot(df=covidMap.set_geometry('centroid'), #set geometry
               scheme=mc.EqualInterval(covidMap.DeathsPer100k,k=3),
               cmap='YlGnBu',#palette
               hue="DeathsPer100k",               
               scale='CasesPer100k', #sizes of points         
               limits=(4, 40), #range for sizes of points         
               legend=True,
               legend_var='hue',
               legend_kwargs={'bbox_to_anchor': (1, 1),
                              'frameon': True,
                              'markeredgecolor':'k',
                              'title':"DeathsPer100k"},
               extent = covidMap.total_bounds,
               ax=countyBorders)
plt.show()


#%%

counties=gplt.polyplot(waCounties,edgecolor= 'silver')

gplt.kdeplot(WApoints,
             shade=False,
             shade_lowest=False,
             ax=counties)


#%% for republicans

counties=gplt.polyplot(waCounties,edgecolor= 'black',
                       projection=gcrs.Mercator())#reproject)

gplt.kdeplot(WApoints[WApoints.party=='REPUBLICAN'], #subset
             shade=True, 
             cmap='Reds',
             shade_lowest=False,
             ax=counties,
             extent=kingMap.total_bounds)


#%% for democrats

counties=gplt.polyplot(waCounties,edgecolor= 'black',
                       projection=gcrs.Mercator())#reproject)
gplt.kdeplot(WApoints[WApoints.party=='DEMOCRAT'], #subset
             shade=True, 
             cmap='Blues',
             shade_lowest=False,
             ax=counties,
             extent=kingMap.total_bounds)


#%%        

import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
from matplotlib_scalebar.scalebar import ScaleBar

plt.figure()
gplt.choropleth(covidMap, 
                hue='CasesPer100k', 
                scheme=mc.HeadTailBreaks(covidMap.CasesPer100k),
                cmap='OrRd',
                legend=True)
#image = plt.imread(cbook.get_sample_data('grace_hopper.png'))
#plt.imshow(image)
scalebar = ScaleBar(1,'m') # 1 pixel = 0.2 meter
plt.gca().add_artist(scalebar)
plt.show()

#%%

fig, ax = plt.subplots(figsize=(10, 10))
gplt.choropleth(covidMap, 
                hue='CasesPer100k', 
                scheme=mc.HeadTailBreaks(covidMap.CasesPer100k),
                cmap='OrRd',
                legend=True,
                legend_kwargs={'bbox_to_anchor': (1, 1),
                              'frameon': True,
                              'markeredgecolor':'k',
                              'title':"DeathsPer100k"},
                extent = covidMap.total_bounds,
                ax=ax)

x, y, arrow_length = 0, 0.3, 0.3
ax.annotate('N', xy=(x, y), 
            xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black',
                            width=5,
                            headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax.transAxes)
plt.show()

