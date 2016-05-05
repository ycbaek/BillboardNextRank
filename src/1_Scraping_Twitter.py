
# coding: utf-8

# In[16]:

import tweepy
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import time
import json
from pymongo import MongoClient


# # Get key words for searching

# In[17]:

def clean_data(df):
    df=df.rename(columns = {'artist':'artist_long'})
    try:
        df["artist"] = df["artist_long"].map(lambda x : x.split('featuring')[0])
    except:
        df["artist"] = df["artist_long"].map(lambda x : x)
    df["ID"] = df["song"] + "%" +  df["artist"]
    col_name = ['date','rank','ID','song','artist']
    df = df[~df['artist'].isnull()]
    df = df[~df['song'].isnull()]
    
    return df[col_name]


# In[18]:

df_raw = pd.read_csv("../data/billboard_result_20160305_20160416.csv")
df=clean_data(df_raw)

# list of key words
Input = [ ["#"+str(id.split('%')[0]), str(id.split('%')[1])]  for id in df["ID"]]


# # Twitter scraping

# In[19]:

def twitterScraping_cursor(query, tab):
    
    for tweet in tweepy.Cursor(api.search, q=query).items():
        tweet_json= tweet._json
        tab.insert(tweet_json)  


# In[9]:

def twitterScraping(query, tab, j):
    
    # After 12 time search, go to next key search
    if j > 12:
        print "It exceeds Max_J and j is {}.".format(j)
        return
    
    try:
        twitterScraping_cursor(query, tab)
        
    except KeyboardInterrupt:
        print "KeyboardInterrupt"
        return
    
    except:
        j += 1
        print "Error encountered at {} on {}, will try in 15 minutes. The curent number of data is {} ".format        (time.asctime(time.localtime()), query[0], tab.count())
        time.sleep(901)
        print "twitter scrape resumes"
        twitterScraping(query, tab, j)
        return


# ## Twitter API credentials

# In[ ]:

#Twitter API credentials
consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_key = os.getenv("TWITTER_ACCESS_TOKEN")
access_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")


# In[ ]:

#authorize twitter, initialize tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


# ## Set Mongo Client and database

# In[ ]:

client = MongoClient('localhost', 27017)
db = client['song5_database']


# ### Run code to start to scrape

# In[ ]:

MaxIt = df.shape[0]

for i in range(0,MaxIt):
    j = 0
    print "Start scraping for id = {}".format(i)
    table_name = "test_01_" + str(i)
    tab = db[table_name]
    query = Input[i]
    continuous_scraper(query, tab, j)
    print "Final number of the data is {} at id = {} among {}".format(tab.count(),i, MaxIt)
    print "---------------------------------------------------"
    
print "Work is completed"

