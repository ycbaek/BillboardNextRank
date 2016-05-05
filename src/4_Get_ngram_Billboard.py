
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

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
    
    return df[col_name]#.drop_duplicates() 


# In[3]:

def remove_duplicates(df_clean):
    df_clean = df_clean[(df_clean['ID'] != "Heartless%KANYE WEST") | (df_clean['date'] != "2009-06-06") | (df_clean['rank'] != 79)]
    
    return df_clean


# In[4]:

def wide_pivot(df_long):
    df= df_long.pivot('ID', 'date', 'rank')
    df["ID"] = df.index
    df["song"] = df["ID"].map(lambda x : x.split('%')[0])
    df["artist"] = df["ID"].map(lambda x : x.split('%')[1])
    df.fillna(101,inplace=True)
    df.columns = [range(len(df.columns)-3)+['ID', 'song', 'artist']]
    #df.index = range(len(df))
    #df["IDN"] = df.index
    return df


# In[5]:

def get_est_form(df, n=4):
    data = []
    for k in xrange(len(df)):
        df_row = df[df.index == df.index[k]]

        for i in xrange(len(df.columns)-n-3):
            ser = [df_row[j].values[0] for j in range(i, i+n+1)]
            if np.product(np.array(ser) <101) == 1:
                diff = []
                for i in range(len(ser)-1):
                    if ser[i+1]-ser[i] > 0:
                        value = -1
                    elif ser[i+1]-ser[i] < 0:
                        value = 1
                    else:
                        value = 0    
                    diff.append(value)
                data.append(diff[:-1] + [ser[n-1]] + [diff[-1]])

    result = pd.DataFrame(data)
    result.columns = [["diff_{}".format(i) for i in range(result.shape[1]-2)] + ['rank','target']]
    file_name = "pattern_{}.csv".format(n)
    result.to_csv(file_name, sep=',', encoding='utf-8')
    return result


# In[6]:

def get_ngram_form(df, n=4, name="../data/ngram"):
    data = []
    for k in xrange(len(df)):
        df_row = df[df.index == df.index[k]]

        for i in xrange(len(df.columns)-n-3):
            ser = [df_row[j].values[0] for j in range(i, i+n+1)]
            ser1 = ser[0:-1]
            if np.product(np.array(ser1) <101) == 1:
                data.append(ser)

    result = pd.DataFrame(data)
    result.columns = [["diff_{}".format(i) for i in range(result.shape[1]-1)] + ['target']]
    file_name = name+"_{}.csv".format(n)
    result.to_csv(file_name, sep=',', encoding='utf-8')
    return result


# # Run code

# ### Get data with type of ngram ( number in ngram = order)

# In[7]:

df_raw = pd.read_csv("../data/billboard_result_19900106_20091226.csv")
#df_raw = pd.read_csv("../data/billboard_result_20100102_20160423.csv")

df_clean = clean_data(df_raw)
df_clean = remove_duplicates(df_clean)
df = wide_pivot(df_clean)

for i in range(1, 11):
    print "i: {}".format(i)
    get_ngram_form(df, n=i, name="../data/ngram_1990_2009")


# In[ ]:



