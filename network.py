#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:04:22 2020

@author: JoseManuel
"""

#%%
import json
import tweepy

# your credentials as a dictionary in a text file:
keysAPI = json.load(open('keysAPI.txt','r'))
api_key = keysAPI['consumer_key']
api_secret_key = keysAPI['consumer_secret']
access_token = keysAPI['access_token']
access_token_secret = keysAPI['access_token_secret']

# authorizing your application:
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)

# some extra attributes
api=tweepy.API(auth, 
               retry_count=3,
               timeout=600,
               wait_on_rate_limit=True,
               wait_on_rate_limit_notify=True,
               parser=tweepy.parsers.JSONParser())



#%%

import pandas as pd

link1="https://github.com/resourcesbookvisual/data"
link2="/raw/master/PresidentsTwitter.xlsx"
LINK=link1 + link2

twusers=pd.read_excel(LINK)

#%%
twusers


#%%

# get timeline
who='realDonaldTrump'
trumpTweets = api.user_timeline(screen_name = who,
                                count = 2,
                                tweet_mode="extended")
# create data frame
dates=[t['created_at'] for t in trumpTweets]
text=[t['full_text'] for t in trumpTweets]
likes=[t['favorite_count'] for t in trumpTweets]
rts=[t['retweet_count'] for t in trumpTweets]

trumpDF=pd.DataFrame({'created_at':dates,
                      'text':text,
                      'retweet_count':rts,
                      'favorite_count':likes})


#%%

from datetime import datetime as dt

trumpDF['created_at']=pd.to_datetime(trumpDF['created_at'],
                                       infer_datetime_format=True) 
trumpDF['Date']=[dt.date(d) for d in trumpDF['created_at']] 
trumpDF['Day']=[dt.date(d).isoweekday() for d in trumpDF['created_at']] 
trumpDF['Hour'] = [dt.time(d).hour for d in trumpDF['created_at']]

# a column for flagging a retweet.
trumpDF['is_retweet'] = trumpDF.text.str.startswith('RT')

# saving the file (commented just to avoid rewriting it)
#trumpDF.to_csv("trumps.csv",index=False) 


#%%

#loading
link3="/raw/master/trumps.csv"
trumpLink=link1 + link3
allTweets=pd.read_csv(trumpLink)

#no retweets
DTtweets=allTweets[~allTweets.is_retweet]
DTtweets.reset_index(drop=True,inplace=True)

#%%
# just readability
dirtyVar=DTtweets.text

# cleaning steps
DTtweets.loc[:,['text']]=dirtyVar.str.replace('[^\x01-\x7F]','')
DTtweets.loc[:,['text']]=dirtyVar.str.replace('http\\S+\\s*','')
DTtweets.loc[:,['text']]=dirtyVar.str.replace('&amp;','and')
DTtweets.loc[:,['text']]=dirtyVar.str.replace('&lt;|&gt;','')

### optional steps
#DTtweets['text']=dirtyVar.str.replace('@\\w+','')
#DTtweets['text']=dirtyVar.str.replace('#\\w+','')


#%%

# punctuation
import string
PUNCs=string.punctuation # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
DTtweets.loc[:,['text']]=dirtyVar.str.replace('['+PUNCs+']', '')

# to lower case
DTtweets.loc[:,['text']]=dirtyVar.str.lower()

#%%


#tokenize into a list
DTwordsList = " ".join(DTtweets.text).split()


#%%

from nltk.corpus import stopwords
STOPS = stopwords.words('english')
DTwordsList=[word for word in DTwordsList if word not in STOPS]


#%%

FTtrump={word:DTwordsList.count(word) for word in DTwordsList}


#%%

import operator
dict(sorted(FTtrump.items(), key=operator.itemgetter(1),reverse=True))

#%%

import matplotlib.pyplot as plt
from wordcloud import WordCloud
 
wc1 = WordCloud(background_color='white')
wc1.generate_from_frequencies(frequencies=FTtrump)
plt.figure()
plt.imshow(wc1, interpolation="bilinear")
plt.axis("off")
plt.show()

#%%

from wordcloud import STOPWORDS

wc1 = WordCloud(background_color='white',
                stopwords=STOPWORDS, #its own list
                collocations=False) # no bigrams
wc1.generate(" ".join(DTtweets.text))
plt.figure()
plt.imshow(wc1, interpolation="bilinear")
plt.axis("off")
plt.show()

#%%

#subsetting
FTsub={k:v for k, v in FTtrump.items() if v>4}

#replotting
wc2 = WordCloud(background_color='white',
                colormap="Reds")
wc2.generate_from_frequencies(frequencies=FTsub)
plt.figure()
plt.imshow(wc2, interpolation="bilinear")
plt.axis("off")
plt.show()


#%%

#recoloring function
def myColor_func(word, **kwargs):
     # key of max value
     kMax=max(FTsub.items(), key=operator.itemgetter(1))[0]
     # 0 for red /  120 for green / 240 for blue
     return "hsl(0, 100%%, %d%%)" % (5*FTsub[kMax]/FTsub[word])

#replotting
wc3 = WordCloud(background_color='white',
                color_func=myColor_func)
wc3.generate_from_frequencies(frequencies=FTsub)
plt.figure()
plt.imshow(wc3, interpolation="bilinear")
plt.axis("off")
plt.show()



#%%

#bigrams per twitter (per cell)
from nltk import bigrams

theBigrams=[bigrams(eachTW.split()) for eachTW in DTtweets.text]


# list of all bigrams
from itertools import chain

pairWords = list(chain(*theBigrams))

pairWords

#%%

# frequency of DTrump bigrams
from collections import Counter

FT_DT_2g_dict = Counter(pairWords) #generate counter

# Turn FT_DT_2g_dict  into dataframe, naming columns
FT_DT_2g = pd.DataFrame(FT_DT_2g_dict.most_common(),
                        columns=['theBigram', 'count'])

#%%

# Turn column of tuples into separate columns
FT_DT_2g['word1'], FT_DT_2g['word2'] = FT_DT_2g.theBigram.str

# Getting rid of stopwords:
FT_DT_2g=FT_DT_2g[~FT_DT_2g['word1'].isin(STOPS)]
FT_DT_2g=FT_DT_2g[~FT_DT_2g['word2'].isin(STOPS)]

FT_DT_2g

#%%

import networkx as nx

# from data frame to graph
DT_2g_net=nx.from_pandas_edgelist(df=FT_DT_2g,
                                  source='word1',
                                  target='word2')
# plotting graph (default layout)
nx.draw_networkx(DT_2g_net, 
                 font_size=7,
                 edge_color='red',
                 node_color='yellow',
                 node_size=100,
                 alpha=0.9,
                 with_labels = True)


#%%
    
#setting size
fig, ax = plt.subplots(figsize=(10,10))

#saving layout positions
pos = nx.spring_layout(DT_2g_net)

# Plot networks
nx.draw_networkx(DT_2g_net, 
                 pos,  #layout
                 edge_color='red',
                 node_color='yellow',
                 node_size=100,
                 with_labels = False,
                 ax=ax) # for matplotlib ax

# labels away from node
for word, freq in pos.items():
    x, y = freq[0]+.01, freq[1]+.01 # new pos values
    ax.text(x, y,#new positions
            s=word,#label
            horizontalalignment='center',
            fontsize=7,rotation=30)
    
plt.show()


#%%

#subsetting
FT_DT_2g3=FT_DT_2g[FT_DT_2g['count']>=3]

DT_2g_net_sub=nx.from_pandas_edgelist(FT_DT_2g3,'word1','word2')

#plotting    
fig, ax = plt.subplots(figsize=(6, 6))
pos = nx.spring_layout(DT_2g_net_sub)

# Plot networks
nx.draw_networkx(DT_2g_net_sub, pos,
                 edge_color='red',node_color='yellow',
                 node_size=100,with_labels = False,ax=ax)

# labels away from node
for word, freq in pos.items():
    x, y = freq[0]+.05, freq[1]+.03
    ax.text(x, y,s=word,horizontalalignment='center', 
            fontsize=13,rotation=30)
    
plt.show()



#%%

#import itertools
#import networkx as nx
#
#net = nx.DiGraph()
#for a,b in itertools.combinations(twusers.twitter,2):
#    res=api.show_friendship(source_screen_name=a,target_screen_name=b)
#    status=res['relationship']['source']['following'], res['relationship']['target']['following']
#    if status[0] and status[1]:
#        net.add_edge(a, b)
#        net.add_edge(b, a)
#    if status[0] and not status[1]:
#        net.add_edge(a, b)
#    if not status[0] and status[1]:
#        net.add_edge(b, a)
#


#%%



#get relationships data
link4="/raw/master/edgesAmericas.csv"
LINK=link1 + link4
relationships=pd.read_csv(LINK)

# build network
import networkx as nx
net=nx.from_pandas_edgelist(relationships,create_using=nx.DiGraph())

# make sure isolates are in the network
net.add_nodes_from(twusers.twitter)

#add attributes from data frame
### data frame as dictionary
attributes=twusers.set_index('twitter').to_dict('index')
### add attributes of nodes to network
nx.set_node_attributes(net, attributes)

#%%

nx.write_graphml_lxml(net, "presiAmericas.graphml") 

#%%

nx.draw_networkx(net)


#%%


pos = nx.nx_pydot.graphviz_layout(net)
nx.draw_networkx(net,pos=pos)


#%%
import matplotlib.pyplot as plt

pos = nx.nx_pydot.graphviz_layout(net)
plt.figure(figsize=(8, 8))
plt.axis('off')
nx.draw_networkx(net, 
                 pos=pos, 
                 with_labels=True, 
                 node_size=25, 
                 edge_color='b')
plt.show()



#%%
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

#layout
pos=nx.spring_layout(net,k=1) #higher k gives more spread of nodes.
#prepare coloring by region
allValues=[n[1]['region'] for n in net.nodes(data=True)]
levels=pd.unique(allValues).tolist()
palette = plt.get_cmap("Set2") #palette

#drawing nodes - coloring nodes per attribute
for categoryChosen in levels:
    #presidents of the same region
    nodesChosen=[node[0] for node in net.nodes(data=True) 
                  if node[1]['region'] in categoryChosen]
    #choosing color for these nodes
    colorChosen=rgb2hex(palette(levels.index(categoryChosen)))  
    # draw chosen nodes
    nx.draw_networkx_nodes(net,pos=pos,node_size=100,
                           node_color=colorChosen,
                           nodelist=nodesChosen,
                           label=categoryChosen) # for legend!
#draw edges
nx.draw_networkx_edges(net,pos=pos,edge_color="silver")

#draw labels for nodes 
#using President name (not Twitter username)
newLabels = {n[0]:n[1]['president'] for n in net.nodes(data=True)}
nx.draw_networkx_labels(net,pos=pos,font_size=8,
                        font_color='grey',
                        font_weight='bold',
                        labels=newLabels)

# requesting legend (needs "label" in nodes above)
plt.legend(markerscale=1, loc="best")
plt.show()


#%%


nx.set_node_attributes(G=net, 
                       values=nx.in_degree_centrality(net),
                       name='indegree')

nx.set_node_attributes(G=net, 
                       values=nx.out_degree_centrality(net),
                       name='outdegree')


#%% node size by continuos attribute

from matplotlib.colors import rgb2hex

plt.figure(figsize=(8,8))
pos=nx.kamada_kawai_layout(net) #layout
#prepare coloring by region
allValues=[n[1]['region'] for n in net.nodes(data=True)]
levels=pd.unique(allValues).tolist()
palette = plt.get_cmap("Set2")

for categoryChosen in levels:
    nodesChosen=[node[0] for node in net.nodes(data=True) 
                  if node[1]['region'] == categoryChosen]
    colorChosen=rgb2hex(palette(levels.index(categoryChosen)))  
    #chosing sizes of the actors (in a list)
    sizesChosen=[(1+x[1]['indegree'])**20 
                 for x in net.nodes(data=True) 
                 if x[0] in list(nodesChosen)]
    # drawing the NODES
    nx.draw_networkx_nodes(net,pos=pos,
                           node_size=sizesChosen, #vector of sizes
                           node_color=colorChosen,
                           nodelist=nodesChosen,
                           label=categoryChosen)
#drawing edges and labels
nx.draw_networkx_edges(net,pos=pos,edge_color="silver")
nx.draw_networkx_labels(net,pos=pos,font_size=8,font_color='grey')

# customizing legend
MyLegend = plt.legend(loc="best")
for handle in MyLegend.legendHandles:
    handle.set_sizes([20.0])

plt.show()

#%% indegree in text size


from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

#setting layout and palette
pos=nx.kamada_kawai_layout(net) 
allValues=[n[1]['region'] for n in net.nodes(data=True)]
levels=pd.unique(allValues).tolist()
palette = plt.get_cmap("Set2")

#drawing nodes and edges
for categoryChosen in levels:
    nodesChosen=[node[0] for node in net.nodes(data=True) 
                  if node[1]['region'] == categoryChosen]
    colorChosen=rgb2hex(palette(levels.index(categoryChosen)))  
    nx.draw_networkx_nodes(net,pos=pos,node_size=100,
                           node_color=colorChosen,
                           nodelist=nodesChosen,
                           label=categoryChosen) 
nx.draw_networkx_edges(net,pos=pos,edge_color="silver")

#drawing each node label - SIZE by indegree!!
for user in net.nodes():
    sizeLabel=net.nodes(data=True)[user]['indegree']+1
    nodeLabel=net.nodes(data=True)[user]['country']
    nx.draw_networkx_labels(net, pos=pos,
                            labels={user:nodeLabel},
                            font_size=sizeLabel**8) #varying sizes
# requesting legend 
# marker in legend same size as actor node
plt.legend(markerscale=1, loc="best")
plt.show()


#%%


from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt #513


plt.figure(figsize=(8,8))
#setting layout and palette
pos=nx.kamada_kawai_layout(net) 
allValues=[n[1]['region'] for n in net.nodes(data=True)]
levels=pd.unique(allValues).tolist()
palette = plt.get_cmap("Set2")

#drawing nodes and edges
for categoryChosen in levels:
    nodesChosen=[node[0] for node in net.nodes(data=True) 
                  if node[1]['region'] == categoryChosen]
    colorChosen=rgb2hex(palette(levels.index(categoryChosen)))  
    nx.draw_networkx_nodes(net,pos=pos,node_size=100,
                           node_color=colorChosen,
                           nodelist=nodesChosen,
                           label=categoryChosen) 
nx.draw_networkx_edges(net,pos=pos,edge_color="silver")

#drawing each node label - SIZE by outdegree!!
for user in net.nodes():
    sizeLabel=net.nodes(data=True)[user]['outdegree']+1
    nodeLabel=net.nodes(data=True)[user]['country']
    nx.draw_networkx_labels(net,pos=pos,
                            labels={user:nodeLabel},
                            font_size=sizeLabel**8)

plt.legend(markerscale=1, loc="best")
plt.show()




#%%

# to indirected
unet=net.to_undirected(reciprocal=True)
# removing isolates
unet.remove_nodes_from(list(nx.isolates(unet)))


#%%

# bring algorithm
from cdlib import algorithms

# Find the communities
modCommunity = algorithms.greedy_modularity(unet).communities


#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

palette = plt.get_cmap("Set1")

pos=nx.kamada_kawai_layout(unet)

# for each community
colorIndex=0
for community in modCommunity:
    # INSTEAD of rgb to hexadecimal: repeating list
    colorChosen=np.tile(palette(colorIndex), (len(community), 1))
    nx.draw_networkx_nodes(unet,pos, 
                           nodelist=community, #nodes chosen
                           node_color=colorChosen)
    colorIndex+=1 #increase index

#edges and labels (default values)    
nx.draw_networkx_edges(unet, pos)
nx.draw_networkx_labels(unet,pos)    
plt.show()

#%%

louvainCommunity = algorithms.louvain(unet).communities

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

palette = plt.get_cmap("Set1")
pos=nx.kamada_kawai_layout(unet)

# for each community
colorIndex=0
for community in louvainCommunity:
    # from rgb to hexadecimal
    chosenColor=rgb2hex(palette(colorIndex))
    nx.draw_networkx_nodes(unet,pos, 
                           nodelist=community, #nodes chosen
                           node_color=chosenColor)
    colorIndex+=1 #increase index

#edges and labels (default values)    
nx.draw_networkx_edges(unet, pos)
nx.draw_networkx_labels(unet,pos)    
plt.show()



#%% evaluation

from cdlib import evaluation
evaluation.newman_girvan_modularity(unet,algorithms.greedy_modularity(unet))
evaluation.newman_girvan_modularity(unet,algorithms.louvain(unet))


#%%

infmpCommunity = algorithms.infomap(net).communities

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

palette = plt.get_cmap("Set1")
pos=nx.kamada_kawai_layout(net)

# for each group
colorIndex=0
for community in infmpCommunity:
    # from rgb to hexadecimal
    chosenColor=rgb2hex(palette(colorIndex))
    nx.draw_networkx_nodes(net,pos, 
                           nodelist=community, #nodes chosen
                           node_color=chosenColor)
    colorIndex+=1 #increase index

#edges and labels (default values)    
nx.draw_networkx_edges(net, pos)
nx.draw_networkx_labels(net,pos)    
plt.show()



#%%

evaluation.newman_girvan_modularity(unet,algorithms.infomap(net))


#%%
import networkx as nx
from cdlib import algorithms, viz

coms = algorithms.infomap(net)
pos = nx.spring_layout(net)
viz.plot_network_clusters(net, coms, pos,plot_labels=True)



#%%
#not in book

from heapq import nlargest
h,a=nx.hits(net)
hubs=nlargest(2, h, key = h.get)

labels = {}    
for node in net.nodes():
    if node in hubs:
        #set the node name as the key and the label as its value 
        labels[node] = node

#%%
# saving the cut point



# positions for all the nodes
pos=nx.kamada_kawai_layout(net,scale=5)

# sizes for nodes
SALIENT, NORMAL=(2000,800)

# plot all nodes
nx.draw(net,pos,node_color='b',
        node_size=NORMAL,with_labels=False, 
        alpha=0.5,node_shape='^')

# make the cut salient:
nx.draw_networkx_nodes(net,pos,nodelist=hubs,
                       node_size=SALIENT,
                       node_color='white')
nx.draw_networkx_labels(net,pos=pos,labels=labels,font_size=7)
plt.show()

#%%

# not in book
from heapq import nlargest
deg=nx.degree_centrality(net)
degs=nlargest(2, deg, key = deg.get)

labels = {}    
for node in net.nodes():
    if node in degs:
        #set the node name as the key and the label as its value 
        labels[node] = node

#%%

# positions for all the nodes
pos=nx.kamada_kawai_layout(net,scale=5)

# sizes for nodes
SALIENT, NORMAL=(100,8)

# plot all nodes
nx.draw_networkx_edges(net,pos=pos,edge_color='red',
                       width=0.5,alpha=0.2 )
nx.draw_networkx_nodes(net,pos,node_shape='^',node_size=NORMAL,
                       with_labels=False)
#nx.draw(net,pos,node_color='b',
#        node_size=NORMAL,with_labels=False, 
#        alpha=0.5,node_shape='^')


# make the cut salient:
nx.draw_networkx_nodes(net,pos,nodelist=degs,
                       node_size=SALIENT,
                       node_color='yellow')
allLabels= nx.draw_networkx_labels(net,pos=pos,labels=labels,font_size=7)
for _,t in allLabels.items():
    t.set_rotation(30)

plt.show()


