
# coding: utf-8

# In[1]:

import pandas as pd
import time
import datetime
from datetime import timedelta
from pymongo import MongoClient
from textblob import TextBlob


# In[2]:

def get_dataframe(tab):
    '''
    Input : table in Mongodb
    Output: pnadas DataFrame
    '''

    dataSet = pd.DataFrame()    
    dataSet["id"] = [tweet['id'] for tweet in tab.find()]
    dataSet["text"] = [tweet['text'] for tweet in tab.find()]
    dataSet["created_at"] = [tweet['created_at'] for tweet in tab.find()]
    dataSet["favorite_count"] = [tweet['favorite_count'] for tweet in tab.find()]
    dataSet["source"] = [tweet['source'] for tweet in tab.find()]
    dataSet["user_id"] = [tweet['user']['id'] for tweet in tab.find()]
    dataSet["user_screen_name"] = [tweet['user']['screen_name'] for tweet in tab.find()]
    dataSet["user_name"] = [tweet['user']['name'] for tweet in tab.find()]
    dataSet["user_created_at"] = [tweet['user']['created_at'] for tweet in tab.find()]
    dataSet["user_description"] = [tweet['user']['description'] for tweet in tab.find()]
    dataSet["user_followers_count"] = [tweet['user']['followers_count'] for tweet in tab.find()]
    dataSet["user_friends_count"] = [tweet['user']['friends_count'] for tweet in tab.find()]
    dataSet["user_location"] = [tweet['user']['location'] for tweet in tab.find()]
    dataSet["user_time_zone"] = [tweet['user']['time_zone'] for tweet in tab.find()]

    return dataSet.drop_duplicates()


# In[49]:

def get_dataframe2(tab1, tab2):
    df1 = get_dataframe(tab1)
    df2 = get_dataframe(tab2)
    df = pd.concat([df1, df2])
    return df.drop_duplicates()
    


# In[50]:

def get_category_sent(score):
    if score > 0 :
        return "Pos"
    elif score < 0:
        return "Neg"
    else:
        return "Neu"


# In[52]:

def find_week_1990(date_time):
    
    day7 = timedelta(days=7)
    date_slected = datetime.date(1990,1,7)
    all_date =[date_slected]
    for i in range(5200):
        date_slected = date_slected + day7
        all_date.append(date_slected)

    week_firstDate = pd.DataFrame()
    week_firstDate['firstData'] = all_date
    week_firstDate['week'] = range(1,5202)
    
    for i in range(5201):
        if date_time.date() < week_firstDate['firstData'][i]:
            return week_firstDate['week'][i]


# In[54]:

def find_date_1990(week_from_one):
    
    day7 = timedelta(days=7)
    day6 = timedelta(days=6)
    date_slected = datetime.date(1990,1,7)
    all_date =[date_slected]
    all_date_last = [date_slected + day6]
    for i in range(5200):
        date_slected = date_slected + day7
        all_date.append(date_slected)
        all_date_last.append(date_slected +day6)

    week_Date = pd.DataFrame()
    week_Date['firstData'] = all_date
    week_Date['lastData'] = all_date_last
    week_Date['week'] = range(1,5202)
        
    index_date = week_from_one - 2
    
    return str(all_date[index_date]) + ' ~ ' + str(all_date_last[index_date])


# In[55]:

def info_twitter_song(tab):

    df= get_dataframe(tab)
    df['sent_score'] = df['text'].map(lambda x : TextBlob(x).sentiment.polarity)
    df['sent_category'] = df['sent_score'].map(get_category_sent)
    df['pos'] = df['sent_category'].map(lambda x: 1 if x == 'Pos' else 0)
    df['neg'] = df['sent_category'].map(lambda x: 1 if x == 'Neg' else 0)
    df['neu'] = df['sent_category'].map(lambda x: 1 if x == 'Neu' else 0)
    df['created_at_time'] = df['created_at'].map(lambda x : datetime.datetime.strptime(x, "%a %b %d %H:%M:%S +0000 %Y"))
    df['created_at_time_min'] = df['created_at_time']
    df['created_at_time_max'] = df['created_at_time']
    df['created_at_time_date'] = df['created_at_time'].map(lambda x : x.date())
    df['week_from_one'] = df['created_at_time'].map(find_week_1990)
    df['count'] = 1

    col_list = ['week_from_one']
    agg_dic = {'favorite_count':sum, 'count':sum, 'pos':sum, 'neg':sum, 'neu':sum,'created_at_time_min': min,               'created_at_time_max': max}
    grouped = df.groupby(col_list).agg(agg_dic)
    grouped = grouped.reset_index()

    grouped['neg_rate']=1.0*grouped['neg']/grouped['count']
    grouped['pos_rate']=1.0*grouped['pos']/grouped['count']
    grouped['neu_rate']=1.0*grouped['neu']/grouped['count']
    grouped['ratio_pos_neg']=1.0*(grouped['pos']+1)/(grouped['neg'] +1)
    grouped['favorite_rate']=1.0*grouped['favorite_count']/grouped['count']

    #selected_col = ['week_from_one','count','pos_rate','neg_rate', 'neu_rate','ratio_pos_neg','favorite_rate']
    #slected = grouped[selected_col]
    slected = grouped
    return slected


# In[56]:

def info_twitter_song2(tab1, tab2):

    df= get_dataframe2(tab1, tab2)
    df['sent_score'] = df['text'].map(lambda x : TextBlob(x).sentiment.polarity)
    df['sent_category'] = df['sent_score'].map(get_category_sent)
    df['pos'] = df['sent_category'].map(lambda x: 1 if x == 'Pos' else 0)
    df['neg'] = df['sent_category'].map(lambda x: 1 if x == 'Neg' else 0)
    df['neu'] = df['sent_category'].map(lambda x: 1 if x == 'Neu' else 0)
    df['created_at_time'] = df['created_at'].map(lambda x : datetime.datetime.strptime(x, "%a %b %d %H:%M:%S +0000 %Y"))
    df['created_at_time_min'] = df['created_at_time']
    df['created_at_time_max'] = df['created_at_time']
    df['created_at_time_date'] = df['created_at_time'].map(lambda x : x.date())
    df['week_from_one'] = df['created_at_time'].map(find_week_1990)
    df['count'] = 1

    col_list = ['week_from_one']
    agg_dic = {'favorite_count':sum, 'count':sum, 'pos':sum, 'neg':sum, 'neu':sum,'created_at_time_min': min,               'created_at_time_max': max}
    grouped = df.groupby(col_list).agg(agg_dic)
    grouped = grouped.reset_index()

    grouped['neg_rate']=1.0*grouped['neg']/grouped['count']
    grouped['pos_rate']=1.0*grouped['pos']/grouped['count']
    grouped['neu_rate']=1.0*grouped['neu']/grouped['count']
    grouped['ratio_pos_neg']=1.0*(grouped['pos']+1)/(grouped['neg'] +1)
    grouped['favorite_rate']=1.0*grouped['favorite_count']/grouped['count']

    #selected_col = ['week_from_one','count','pos_rate','neg_rate', 'neu_rate','ratio_pos_neg','favorite_rate']
    #slected = grouped[selected_col]
    slected = grouped
    return slected
    
    


# In[57]:

def info_raw_twitter_song2(song_id,tab1, tab2, list_input):

    df= get_dataframe2(tab1, tab2)
    df['sent_score'] = df['text'].map(lambda x : TextBlob(x).sentiment.polarity)
    df['sent_category'] = df['sent_score'].map(get_category_sent)
    df['pos'] = df['sent_category'].map(lambda x: 1 if x == 'Pos' else 0)
    df['neg'] = df['sent_category'].map(lambda x: 1 if x == 'Neg' else 0)
    df['neu'] = df['sent_category'].map(lambda x: 1 if x == 'Neu' else 0)
    df['created_at_time'] = df['created_at'].map(lambda x : datetime.datetime.strptime(x, "%a %b %d %H:%M:%S +0000 %Y"))
    df['created_at_time_date'] = df['created_at_time'].map(lambda x : x.date())
    df['week_from_one'] = df['created_at_time'].map(find_week_1990)
    df['Song_ID'] = list_input["ID"].values[song_id]

    selected_col = ["Song_ID", "created_at_time","created_at_time_date","week_from_one", "favorite_count",                    "sent_score","sent_category", "pos","neg", "neu"]
    
    return df[selected_col]


# In[58]:

def remove_duplicates(df_clean):
    df_clean = df_clean[(df_clean['ID'] != "Heartless%KANYE WEST") | (df_clean['date'] != "2009-06-06") | (df_clean['rank'] != 79)]
    
    return df_clean


# In[59]:

def get_billboard(billboard_filename_path):

    df2 = pd.read_csv(billboard_filename_path)
    df2 = df2.drop("Unnamed: 0", axis = 1)
    df2.rename(columns={'artist': 'long_artist'}, inplace=True)
    
    df2['last_date_time'] = df2['date'].map(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d"))
    df2['week_from_one'] = df2['last_date_time'].map(find_week_1990)
    df2["artist"] = df2["long_artist"].map(lambda x : x.split('featuring')[0])
    df2["ID"] = df2["song"] + "%" +  df2["artist"]
    df2 = remove_duplicates(df2)
    
    df= df2.pivot('ID', 'week_from_one', 'rank')
    df["ID"] = df.index
    df["song"] = df["ID"].map(lambda x : x.split('%')[0])
    df["artist"] = df["ID"].map(lambda x : x.split('%')[1])
    df.fillna(101,inplace=True)
    df.index = range(1,len(df)+1)
    df["IDN"] = df.index
    
    return df


# In[60]:

def list_no_twitter(list_id, db):
    list_count = []
    for song_id in list_id:
        table_name = "test_01_" + str(song_id)
        tab = db[table_name]
        list_count.append(tab.find().count())
    
    return list(np.array(list_id)[np.array(list_count) == 0])


# In[61]:

def list_no_twitter2(list_id, db1, db2):
    list1 = list_no_twitter(list_id, db1)
    list2 = list_no_twitter(list_id, db2)
    
    return list(set(list1).intersection(set(list2)))


# In[62]:

def twitter_info_table(song_id, db, list_input, current_billboard, week_all):

    table_name = "test_01_" + str(song_id)
    tab = db[table_name]
    
    Input = [ ["#"+str(id.split('%')[0]), str(id.split('%')[1])]  for id in list_input["ID"]]
    max_week = max([x for x in list(current_billboard.columns) if isinstance(x, int)])
    current_billboard[max_week+1] = 0
    Song_history = current_billboard[current_billboard['ID']==list_input['ID'].values[song_id]]

    twit = info_twitter_song(tab)
    twit['diff_time'] = twit['created_at_time_max'] - twit['created_at_time_min']
    twit['diff_hour'] = twit['diff_time'].astype(pd.Timedelta).map(lambda x : float(x.seconds)/3600)
    twit['diff_hour_adj'] = twit['diff_hour'].map(lambda x: 24*7 if x == 0 else x )
    twit['twitter_per_hour'] = 1.0*twit['count']/twit['diff_hour_adj']
    
    
    col = ['week_from_one', 'twitter_per_hour','pos_rate', 'neg_rate','ratio_pos_neg' ,'favorite_rate']
    twitter_table = twit[col]
    
    diff = set(week_all) - set(list(twitter_table['week_from_one'].values))
    
    if len(diff) > 0:
        for i in diff:
            add_one = twitter_table.copy().iloc[0:1,]
            add_one['week_from_one'] = i
            add_one['twitter_per_hour'] = 1.0/(24*7)
            add_one['pos_rate'] = 0
            add_one['neg_rate'] = 0
            add_one['ratio_pos_neg'] = 1
            add_one['favorite_rate'] = 0
            twitter_table = pd.concat([add_one, twitter_table])
            
    twitter_table['Billboard_rank'] = [int(Song_history[week]) for week in twitter_table['week_from_one']]
    twitter_table['date_period'] = twitter_table['week_from_one'].map(find_date_1990)
    twitter_table['Song_ID'] = list_input["ID"].values[song_id]
    twitter_table['Song_IDN'] = song_id+1
    
    
    return Song_history, twitter_table


# In[63]:

def twitter_info_table2(song_id, db1, db2, list_input, current_billboard, week_all):

    table_name = "test_01_" + str(song_id)
    tab1 = db1[table_name]
    tab2 = db2[table_name]
    
    Input = [ ["#"+str(id.split('%')[0]), str(id.split('%')[1])]  for id in list_input["ID"]]
    max_week = max([x for x in list(current_billboard.columns) if isinstance(x, int)])
    current_billboard[max_week+1] = 0
    Song_history = current_billboard[current_billboard['ID']==list_input['ID'].values[song_id]]

    twit = info_twitter_song2(tab1, tab2)
    twit['diff_time'] = twit['created_at_time_max'] - twit['created_at_time_min']
    twit['diff_hour'] = twit['diff_time'].astype(pd.Timedelta).map(lambda x : float(x.seconds)/3600)
    twit['diff_hour_adj'] = twit['diff_hour'].map(lambda x: 24*7 if x == 0 else x )
    twit['twitter_per_hour'] = 1.0*twit['count']/twit['diff_hour_adj']
    
    
    col = ['week_from_one', 'twitter_per_hour','pos_rate', 'neg_rate','ratio_pos_neg' ,'favorite_rate']
    twitter_table = twit[col]
    
    diff = set(week_all) - set(list(twitter_table['week_from_one'].values))
    
    if len(diff) > 0:
        for i in diff:
            add_one = twitter_table.copy().iloc[0:1,]
            add_one['week_from_one'] = i
            add_one['twitter_per_hour'] = 1.0/(24*7)
            add_one['pos_rate'] = 0
            add_one['neg_rate'] = 0
            add_one['ratio_pos_neg'] = 1
            add_one['favorite_rate'] = 0
            twitter_table = pd.concat([add_one, twitter_table])
            
    twitter_table['Billboard_rank'] = [int(Song_history[week]) for week in twitter_table['week_from_one']]
    twitter_table['date_period'] = twitter_table['week_from_one'].map(find_date_1990)
    twitter_table['Song_ID'] = list_input["ID"].values[song_id]
    twitter_table['Song_IDN'] = song_id+1
    
    
    return Song_history, twitter_table


# In[64]:

def twitter_info_table_no_twitter(song_id, list_input, current_billboard, week_all):


    Input = [ ["#"+str(id.split('%')[0]), str(id.split('%')[1])]  for id in list_input["ID"]]
    max_week = max([x for x in list(current_billboard.columns) if isinstance(x, int)])
    current_billboard[max_week+1] = 0
    
    Song_history = current_billboard[current_billboard['ID']==list_input['ID'].values[song_id]]

    
    twitter_table = pd.DataFrame()
    twitter_table['week_from_one'] = week_all
    twitter_table['twitter_per_hour'] = 1.0/(24*7)
    twitter_table['pos_rate'] = 0
    twitter_table['neg_rate'] = 0
    twitter_table['ratio_pos_neg'] = 1
    twitter_table['favorite_rate'] = 0
            
    twitter_table['Billboard_rank'] = [int(Song_history[week]) for week in twitter_table['week_from_one']]
    twitter_table['date_period'] = twitter_table['week_from_one'].map(find_date_1990)
    #twitter_table['Billboard_rank_text'] = twitter_table['Billboard_rank'].map(lambda x : x if x <= 100 else "Not on Billboard")
    twitter_table['Song_ID'] = list_input["ID"].values[song_id]
    twitter_table['Song_IDN'] = song_id+1
    
    return Song_history, twitter_table


# In[65]:

def get_DataFrame_for_song(song_id, db, list_input, current_billboard, week_all):
    
    NoTwitter = list_no_twitter(range(len(list_input)), db)
    
    if song_id in NoTwitter:
        Song_history, twitter_table = twitter_info_table_no_twitter(song_id, db, list_input, current_billboard, week_all)
    else:
        Song_history, twitter_table = twitter_info_table(song_id, db, list_input, current_billboard, week_all)
    
    df_song = twitter_table
    df_song = df_song.rename(columns = {'week_from_one':'week'})
    df_song = df_song.rename(columns = {'Billboard_rank':'current_rank'})
    df_song['past_rank_1'] = df_song['week'].map(lambda x : int(Song_history[x-1]))
    df_song['past_rank_2'] = df_song['week'].map(lambda x : int(Song_history[x-2]))
    df_song['past_rank_3'] = df_song['week'].map(lambda x : int(Song_history[x-3]))
    df_song['past_rank_4'] = df_song['week'].map(lambda x : int(Song_history[x-4]))
    df_song['past_rank_5'] = df_song['week'].map(lambda x : int(Song_history[x-5]))
    df_song['past_rank_6'] = df_song['week'].map(lambda x : int(Song_history[x-6]))
    df_song['past_rank_7'] = df_song['week'].map(lambda x : int(Song_history[x-7]))
    df_song['past_rank_8'] = df_song['week'].map(lambda x : int(Song_history[x-8]))
    df_song['past_rank_9'] = df_song['week'].map(lambda x : int(Song_history[x-9]))
    df_song['next_rank'] = df_song['week'].map(lambda x : int(Song_history[x+1]))                                     
    
    
    col =['date_period','week','current_rank','count','twitter_per_hour','pos_rate',          'neg_rate','ratio_pos_neg','favorite_rate','Song_ID','Song_IDN',          'past_rank_1','past_rank_2','past_rank_3','past_rank_4','past_rank_5',          'past_rank_6','past_rank_7','past_rank_8','past_rank_9''next_rank']
    
    return df_song
    


# In[66]:

def get_DataFrame_for_song2(song_id, db1, db2, list_input, current_billboard, week_all):
    
    NoTwitter = list_no_twitter2(range(len(list_input)), db1, db2)
    
    if song_id in NoTwitter:
        Song_history, twitter_table = twitter_info_table_no_twitter(song_id, list_input, current_billboard, week_all)
    else:
        Song_history, twitter_table = twitter_info_table2(song_id, db1, db2, list_input, current_billboard, week_all)
    
    df_song = twitter_table
    df_song = df_song.rename(columns = {'week_from_one':'week'})
    df_song = df_song.rename(columns = {'Billboard_rank':'current_rank'})
    df_song['past_rank_1'] = df_song['week'].map(lambda x : int(Song_history[x-1]))
    df_song['past_rank_2'] = df_song['week'].map(lambda x : int(Song_history[x-2]))
    df_song['past_rank_3'] = df_song['week'].map(lambda x : int(Song_history[x-3]))
    df_song['past_rank_4'] = df_song['week'].map(lambda x : int(Song_history[x-4]))
    df_song['past_rank_5'] = df_song['week'].map(lambda x : int(Song_history[x-5]))
    df_song['past_rank_6'] = df_song['week'].map(lambda x : int(Song_history[x-6]))
    df_song['past_rank_7'] = df_song['week'].map(lambda x : int(Song_history[x-7]))
    df_song['past_rank_8'] = df_song['week'].map(lambda x : int(Song_history[x-8]))
    df_song['past_rank_9'] = df_song['week'].map(lambda x : int(Song_history[x-9]))
    df_song['next_rank'] = df_song['week'].map(lambda x : int(Song_history[x+1]))                                     
    
    
    col =['date_period','week','current_rank','count','twitter_per_hour','pos_rate',          'neg_rate','ratio_pos_neg','favorite_rate','Song_ID','Song_IDN',          'past_rank_1','past_rank_2','past_rank_3','past_rank_4','past_rank_5',          'past_rank_6','past_rank_7','past_rank_8','past_rank_9''next_rank']
    
    return df_song
    


# In[67]:

def merge_dataframe(list_id, db, list_input, current_billboard, file_name, week_all=[]):
    
    print "We are getting data frame for id = {}".format(list_id[0])
    df = get_DataFrame_for_song(list_id[0], db, list_input, current_billboard, week_all)    
    print "The length of dataFrame is {}.".format(len(df))
    print " "
    
    if len(list_id) == 1:
        print "We're done"
        return df
    
    
    for i in range(1, len(list_id)):
        print "We completed {} of the total work and now are merging the data frame with id = {}".format        (1.0*i/len(list_id), list_id[i])
        
        add_df = get_DataFrame_for_song(list_id[i], db, list_input, current_billboard, week_all)
        df = pd.concat([df, add_df])
        print "The length of dataFrame is {}.".format(len(df))
        print " "
    
    df.to_csv(file_name, sep=',', encoding='utf-8')
    print "We're done"
    return df
    


# In[68]:

def merge_dataframe2(list_id, db1, db2, list_input, current_billboard, file_name, week_all=[]):
    
    print "We are getting data frame for id = {}".format(list_id[0])
    df = get_DataFrame_for_song2(list_id[0], db1, db2, list_input, current_billboard, week_all)    
    print "The length of dataFrame is {}.".format(len(df))
    print " "
    
    if len(list_id) == 1:
        print "We're done"
        return df
    
    
    for i in range(1, len(list_id)):
        print "We completed {} of the total work and now are merging the data frame with id = {}".format        (1.0*i/len(list_id), list_id[i])
        
        add_df = get_DataFrame_for_song2(list_id[i], db1, db2, list_input, current_billboard, week_all)
        df = pd.concat([df, add_df])
        print "The length of dataFrame is {}.".format(len(df))
        print " "
    
    df.to_csv(file_name, sep=',', encoding='utf-8')
    print "We're done"
    return df
    


# In[69]:

def merge_raw_dataframe2(list_id, db1, db2, list_input, file_name):
    
    print "We are getting data frame for id = {}".format(list_id[0])
    table_name = "test_01_" + str(list_id[0])
    tab1 = db1[table_name]
    tab2 = db2[table_name]
    df = info_raw_twitter_song2(list_id[0],tab1, tab2, list_input)
    print "The length of dataFrame is {}.".format(len(df))
    print " "
    
    if len(list_id) == 1:
        print "We're done"
        return df
    
    
    for i in range(1, len(list_id)):
        print "We completed {} of the total work and now are merging the data frame with id = {}".format        (1.0*i/len(list_id), list_id[i])
        
        table_name = "test_01_" + str(list_id[i])
        tab1 = db1[table_name]
        tab2 = db2[table_name]
        add_df = info_raw_twitter_song2(list_id[i],tab1, tab2, list_input)
        df = pd.concat([df, add_df])
        print "The length of dataFrame is {}.".format(len(df))
        print " "
    
    df.to_csv(file_name, sep=',', encoding='utf-8')
    print "We're done"
    return df


# In[ ]:




# # Run Code

# In[70]:

client = MongoClient('localhost', 27017)

db1 = client['song3_database']
db2 = client['song4_database']
list_input = get_billboard("../data/billboard_result_20160305_20160416.csv")
current_billboard = get_billboard("../data/billboard_result_20150103_20160430.csv")
week_1371_1373 = [1371,1372,1373]


# ## test example

# In[ ]:

list_id = range(2)

file_name = '../data/result_song_3_4_temp.csv' 
colect_song = merge_dataframe2(list_id, db1, db2, list_input, current_billboard, file_name, week_all=week_1371_1373)

file_name_raw = '../data_result_raw_song_3_4_temp.csv' 
df_raw_total = merge_raw_dataframe2(list_id, db1, db2, file_name_raw)


# ## Run whole set  

# In[71]:

list_id = range(len(list_input))


# In[72]:

file_name = '../data/result_database34_20160430.csv' 
colect_song = merge_dataframe2(list_id, db1, db2, list_input, current_billboard, file_name, week_all=week_1371_1373)


# In[ ]:




# In[ ]:




# In[73]:

file_name_raw = '../data/result_raw__database34_20160430.csv' 
df_raw_total = merge_raw_dataframe2(list_id, db1, db2, list_input, file_name_raw)


# In[74]:

1+1


# In[ ]:




# In[ ]:



