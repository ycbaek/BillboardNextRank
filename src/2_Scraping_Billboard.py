
# coding: utf-8

# In[1]:

import requests
import datetime
import time
from datetime import timedelta
from bs4 import BeautifulSoup, Comment


# In[24]:

def get_url(date):
    '''
    For given data, retuen url for Billboard Hot 100 Chart
    '''
    
    str_date=''.join([date[0:4], date[5:7], date[8:10]])
    url = "http://www.umdmusic.com/default.asp?Lang=English&Chart=D&ChDate={}&ChMode=P".format(str_date)
    return url


# In[25]:

def get_song_artist(song_raw_data):
    
    '''
    Input : song name, space, space,.. ,space , artist name
    Output : song name, artist name
    
    '''
    song_name = []
    artist_name = []
    space_num = 0
    for item in song_raw_data.split(" "):
        if len(item) == 0:
            space_num +=1
        if space_num == 0:
            song_name.append(item)
        else:
            if len(item) !=0:
                artist_name.append(item)
    return " ".join(song_name), " ".join(artist_name)


# In[26]:

def get_song_info(link):
    '''
    Input :  information of URL
    Output : song name, artist name
    '''
    set_info = link.findAll('td')
    rank = set_info[0].text
    song, artist = get_song_artist(set_info[4].text) 
    return rank, song, artist


# In[27]:

def get_all_info_one_week (data):
    
    '''
    Input : date
    Output : date, rank, song, artist
    '''
    url =  get_url(data.strftime('%Y-%m-%d'))
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    AllComments = soup.findAll(text=lambda text:isinstance(text,Comment))
    index_num = AllComments.index(' Display Chart Table ')
    RankTable = AllComments[index_num].find_next_sibling('table')
    RankList = RankTable.findAll("tr")
    
    ListD = []
    ListR = []
    ListS = []
    ListA = []

    for i, link in enumerate(RankList):
        if i >= 2:
            rank, song, artist = get_song_info(link)
            ListD =data.strftime('%Y-%m-%d')
            ListR.append(rank)
            ListS.append(song)
            ListA.append(artist)

    table = pd.DataFrame({'date':ListD, 'rank':ListR, 'song':ListS, 'artist':ListA})
    return table
    


# In[6]:

def get_all_info(start_date = datetime.datetime.strptime('2016-01-02', "%Y-%m-%d"),                 end_date = datetime.datetime.fromtimestamp(time.time())):
    '''
    Input : start_date, end_date
    Output : list of date, rank, song, and artist
    '''

    next_date = start_date
    table = get_all_info_one_week (next_date)
    while next_date +  timedelta(days=7) <= end_date:
        next_date = next_date +  timedelta(days=7)
        table = pd.concat([table, get_all_info_one_week (next_date)])
    
    filename = "../data/billboard_result_{}_{}.csv".format(start_date.strftime('%Y%m%d'), next_date.strftime('%Y%m%d'))
    table.to_csv(filename, sep=',', encoding='utf-8')
    return table


# # Run code

# In[ ]:

startDate = datetime.datetime.strptime('2016-03-05', "%Y-%m-%d")
endDate = datetime.datetime.strptime('2016-04-30', "%Y-%m-%d")
get_all_info(start_date=startDate, end_date=endDate)

