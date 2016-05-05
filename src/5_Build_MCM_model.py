
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[2]:

def get_base(dfngram, ngram = 2):
    dfngram = dfngram.astype('int')
    col = ["diff_{}".format(i) for i in range(ngram)]
    df = dfngram[col]
    ASET = set()
    for i in xrange(len(df)):
        ASET.add(tuple(df.iloc[i]))
    
    return ASET


# In[3]:

def get_markov(dfngram, ngram = 2):
    
    if ngram == 1:
        base = set(range(1,101))
    else:
        base = get_base(dfngram, ngram = ngram)
        
    xy_table = pd.DataFrame(data = np.zeros([len(base), 101]), index=base, columns=range(1,102))
    x_table = pd.DataFrame(data = np.zeros([len(base), 1]), index=base, columns= ['target'])
    
    dfngram = dfngram.astype('int')
    col = ["diff_{}".format(i) for i in range(ngram)]
    df = dfngram[col]
    target = dfngram['target'].values
    
    for i in range(len(df)):
        if ngram == 1:
            t = df.iloc[i].values[0]
        else:
            t = tuple(df.iloc[i])
            
        s = target[i]
        xy_table.ix[t,s] += 1
        x_table.ix[t,"target"] += 1
    
    prob_value = np.array(xy_table)/np.array(x_table)
    p_table = pd.DataFrame(data = prob_value, index=base, columns=range(1,102))
    
    p_table["base"] = p_table.index
    p_xy = p_table[range(1,102)].values
    p_y = np.array(range(1,102))
    p_table['pred'] = np.dot(p_xy,p_y)
    
    #return p_table, xy_table, x_table, base
    return p_table, base
    


# In[4]:

def get_mse_7(dfngram, p_table7, p_table6, p_table5, p_table4, p_table3, p_table2, p_table1,               base7, base6, base5, base4, base3, base2, base1):
    
    ngram = 7
    dfngram = dfngram.astype('int')
    col = ["diff_{}".format(i) for i in range(ngram)]
    df = dfngram[col]

    est_list = []
    diff_list = []
    for i in range(len(df)):
        basei = tuple(df.iloc[i])
        esti = get_est_7(basei, p_table7, p_table6, p_table5, p_table4, p_table3, p_table2, p_table1,               base7, base6, base5, base4, base3, base2, base1)
        est_list.append(esti)
        diff_list.append(dfngram['target'].values[i] - esti)
    
    dfngram['est'] = est_list
    dfngram['diff'] = diff_list
    mse_list = np.array(diff_list)
    return np.sqrt(np.mean(mse_list**2))


# In[5]:

def get_mse_6(dfngram, p_table6, p_table5, p_table4, p_table3, p_table2, p_table1,              base6, base5, base4, base3, base2, base1):
    
    ngram = 6
    dfngram = dfngram.astype('int')
    col = ["diff_{}".format(i) for i in range(ngram)]
    df = dfngram[col]

    est_list = []
    diff_list = []
    for i in range(len(df)):
        basei = tuple(df.iloc[i])
        esti = get_est_6(basei, p_table6, p_table5, p_table4, p_table3, p_table2, p_table1,              base6, base5, base4, base3, base2, base1)
        est_list.append(esti)
        diff_list.append(dfngram['target'].values[i] - esti)
    
    dfngram['est'] = est_list
    dfngram['diff'] = diff_list
    mse_list = np.array(diff_list)
    
    return np.sqrt(np.mean(mse_list**2))


# In[6]:

def get_mse_5(dfngram, p_table5, p_table4, p_table3, p_table2, p_table1,              base5, base4, base3, base2, base1):
    
    ngram = 5
    dfngram = dfngram.astype('int')
    col = ["diff_{}".format(i) for i in range(ngram)]
    df = dfngram[col]

    est_list = []
    diff_list = []
    for i in range(len(df)):
        basei = tuple(df.iloc[i])
        esti = get_est_5(basei, p_table5, p_table4, p_table3, p_table2, p_table1,              base5, base4, base3, base2, base1)
        est_list.append(esti)
        diff_list.append(dfngram['target'].values[i] - esti)
    
    dfngram['est'] = est_list
    dfngram['diff'] = diff_list
    mse_list = np.array(diff_list)
    
    return np.sqrt(np.mean(mse_list**2))


# In[7]:

def get_mse_4(dfngram, p_table4, p_table3, p_table2, p_table1, base4, base3, base2, base1):
    
    ngram = 4
    dfngram = dfngram.astype('int')
    col = ["diff_{}".format(i) for i in range(ngram)]
    df = dfngram[col]

    est_list = []
    diff_list = []
    for i in range(len(df)):
        basei = tuple(df.iloc[i])
        esti = get_est_4(basei, p_table4, p_table3, p_table2, p_table1, base4, base3, base2, base1)
        est_list.append(esti)
        diff_list.append(dfngram['target'].values[i] - esti)
    
    dfngram['est'] = est_list
    dfngram['diff'] = diff_list
    mse_list = np.array(diff_list)
    
    return np.sqrt(np.mean(mse_list**2))


# In[8]:

def get_mse_3(dfngram, p_table3, p_table2, p_table1, base3, base2, base1):
    
    ngram = 3
    dfngram = dfngram.astype('int')
    col = ["diff_{}".format(i) for i in range(ngram)]
    df = dfngram[col]

    est_list = []
    diff_list = []
    for i in range(len(df)):
        basei = tuple(df.iloc[i])
        esti = get_est_3(basei, p_table3, p_table2, p_table1, base3, base2, base1)
        est_list.append(esti)
        diff_list.append(dfngram['target'].values[i] - esti)
    
    dfngram['est'] = est_list
    dfngram['diff'] = diff_list
    mse_list = np.array(diff_list)
    
    return np.sqrt(np.mean(mse_list**2))


# In[9]:

def get_mse_2(dfngram, p_table2, p_table1, base2, base1):
    
    ngram = 2
    dfngram = dfngram.astype('int')
    col = ["diff_{}".format(i) for i in range(ngram)]
    df = dfngram[col]

    est_list = []
    diff_list = []
    for i in range(len(df)):
        basei = tuple(df.iloc[i])
        esti = get_est_2(basei, p_table2, p_table1, base2, base1)
        est_list.append(esti)
        diff_list.append(dfngram['target'].values[i] - esti)
    
    dfngram['est'] = est_list
    dfngram['diff'] = diff_list
    mse_list = np.array(diff_list)
    
    return np.sqrt(np.mean(mse_list**2))


# In[10]:

def get_mse_1(dfngram, p_table1, base1):
    
    ngram = 1
    dfngram = dfngram.astype('int')
    col = ["diff_{}".format(i) for i in range(ngram)]
    df = dfngram[col]

    est_list = []
    diff_list = []
    for i in range(len(df)):
        basei = tuple(df.iloc[i])
        esti = get_est_1(basei, p_table1, base1)
        est_list.append(esti)
        diff_list.append(dfngram['target'].values[i] - esti)
    
    dfngram['est'] = est_list
    dfngram['diff'] = diff_list
    mse_list = np.array(diff_list)
    
    return np.sqrt(np.mean(mse_list**2))


# In[11]:

def get_est_7(basei, p_table7, p_table6, p_table5, p_table4, p_table3, p_table2, p_table1,               base7, base6, base5, base4, base3, base2, base1):
    
    if basei in base7:
        return p_table7[p_table7['base']==basei]['pred'].values[0]
    else:
        if basei[1:] in base6:
            return p_table6[p_table6['base']==basei[1:]]['pred'].values[0]
        else:
            if basei[2:] in base5:
                return p_table5[p_table5['base']==basei[2:]]['pred'].values[0]
            else: 
                if basei[3:] in base4:
                    return p_table4[p_table4['base']==basei[3:]]['pred'].values[0]
                else:
                    if basei[4:] in base3:
                        return p_table3[p_table3['base']==basei[4:]]['pred'].values[0]
                    else:
                        if basei[5:] in base2:
                            return p_table2[p_table2['base']==basei[5:]]['pred'].values[0]
                        else:
                            return p_table1[p_table1['base']==basei[6]]['pred'].values[0]


# In[12]:

def get_est_6(basei, p_table6, p_table5, p_table4, p_table3, p_table2, p_table1,              base6, base5, base4, base3, base2, base1):
    
    if basei in base6:
        return p_table6[p_table6['base']==basei]['pred'].values[0]
    else:
        if basei[1:] in base5:
            return p_table5[p_table5['base']==basei[1:]]['pred'].values[0]
        else:
            if basei[2:] in base4:
                return p_table4[p_table4['base']==basei[2:]]['pred'].values[0]
            else: 
                if basei[3:] in base3:
                    return p_table3[p_table3['base']==basei[3:]]['pred'].values[0]
                else:
                    if basei[4:] in base2:
                        return p_table2[p_table2['base']==basei[4:]]['pred'].values[0]
                    else:
                        return p_table1[p_table1['base']==basei[5]]['pred'].values[0]


# In[13]:

def get_est_5(basei, p_table5, p_table4, p_table3, p_table2, p_table1,              base5, base4, base3, base2, base1):
    
    if basei in base5:
        return p_table5[p_table5['base']==basei]['pred'].values[0]
    else:
        if basei[1:] in base4:
            return p_table4[p_table4['base']==basei[1:]]['pred'].values[0]
        else:
            if basei[2:] in base3:
                return p_table3[p_table3['base']==basei[2:]]['pred'].values[0]
            else: 
                if basei[3:] in base2:
                    return p_table2[p_table2['base']==basei[3:]]['pred'].values[0]
                else:
                    return p_table1[p_table1['base']==basei[4]]['pred'].values[0]


# In[14]:

def get_est_4(basei, p_table4, p_table3, p_table2, p_table1, base4, base3, base2, base1):
    
    if basei in base4:
        return p_table4[p_table4['base']==basei]['pred'].values[0]
    else:
        if basei[1:] in base3:
            return p_table3[p_table3['base']==basei[1:]]['pred'].values[0]
        else:
            if basei[2:] in base2:
                return p_table2[p_table2['base']==basei[2:]]['pred'].values[0]
            else: 
                return p_table1[p_table1['base']==basei[3]]['pred'].values[0]


# In[15]:

def get_est_3(basei, p_table3, p_table2, p_table1, base3, base2, base1):
    
    if basei in base3:
        return p_table3[p_table3['base']==basei]['pred'].values[0]
    else:
        if basei[1:] in base2:
            return p_table2[p_table2['base']==basei[1:]]['pred'].values[0]
        else:
            return p_table1[p_table1['base']==basei[2]]['pred'].values[0]


# In[16]:

def get_est_2(basei, p_table2, p_table1, base2, base1):
    
    if basei in base2:
        return p_table2[p_table2['base']==basei]['pred'].values[0]
    else:
        return p_table1[p_table1['base']==basei[1]]['pred'].values[0]


# In[17]:

def get_est_1(basei, p_table1, base1):
    
    basei = basei[0]
    return p_table1[p_table1['base']==basei]['pred'].values[0]
    


# In[18]:

def get_train_test(dfngram1_train, dfngram2_train, dfngram3_train, dfngram4_train, dfngram5_train,                   dfngram6_train, dfngram7_train, dfngram1_test, dfngram2_test, dfngram3_test,                   dfngram4_test, dfngram5_test, dfngram6_test, dfngram7_test, file_name = "../data/temp.csv"):
    

            
    p_table_train7, base_train7 = get_markov(dfngram7_train, ngram = 7)
    p_table_train6, base_train6 = get_markov(dfngram6_train, ngram = 6)
    p_table_train5, base_train5 = get_markov(dfngram5_train, ngram = 5)
    p_table_train4, base_train4 = get_markov(dfngram4_train, ngram = 4)
    p_table_train3, base_train3 = get_markov(dfngram3_train, ngram = 3)
    p_table_train2, base_train2 = get_markov(dfngram2_train, ngram = 2)
    p_table_train1, base_train1 = get_markov(dfngram1_train, ngram = 1)
    
        
    mse_train7 = get_mse_7(dfngram7_train, p_table_train7, p_table_train6, p_table_train5, p_table_train4,                           p_table_train3, p_table_train2, p_table_train1,                           base_train7, base_train6, base_train5, base_train4,                           base_train3, base_train2, base_train1)
    mse_test7 = get_mse_7(dfngram7_test, p_table_train7, p_table_train6, p_table_train5, p_table_train4,                           p_table_train3, p_table_train2, p_table_train1,                           base_train7, base_train6, base_train5, base_train4,                           base_train3, base_train2, base_train1)

    mse_train6 = get_mse_6(dfngram6_train, p_table_train6, p_table_train5, p_table_train4,                           p_table_train3, p_table_train2, p_table_train1,                           base_train6, base_train5, base_train4,                           base_train3, base_train2, base_train1)
    mse_test6 = get_mse_6(dfngram6_test, p_table_train6, p_table_train5, p_table_train4,                           p_table_train3, p_table_train2, p_table_train1,                           base_train6, base_train5, base_train4,                           base_train3, base_train2, base_train1)

    mse_train5 = get_mse_5(dfngram5_train, p_table_train5, p_table_train4,                           p_table_train3, p_table_train2, p_table_train1,                           base_train5, base_train4,                           base_train3, base_train2, base_train1)
    mse_test5 = get_mse_5(dfngram5_test, p_table_train5, p_table_train4,                           p_table_train3, p_table_train2, p_table_train1,                           base_train5, base_train4,                           base_train3, base_train2, base_train1)

    mse_train4 = get_mse_4(dfngram4_train, p_table_train4, p_table_train3, p_table_train2, p_table_train1,                           base_train4, base_train3, base_train2, base_train1)
    mse_test4 = get_mse_4(dfngram4_test, p_table_train4, p_table_train3, p_table_train2, p_table_train1,                           base_train4, base_train3, base_train2, base_train1)

    mse_train3 = get_mse_3(dfngram3_train, p_table_train3, p_table_train2, p_table_train1,                           base_train3, base_train2, base_train1)
    mse_test3 = get_mse_3(dfngram3_test, p_table_train3, p_table_train2, p_table_train1,                           base_train3, base_train2, base_train1)

    mse_train2 = get_mse_2(dfngram2_train, p_table_train2, p_table_train1, base_train2, base_train1)
    mse_test2 = get_mse_2(dfngram2_test, p_table_train2, p_table_train1, base_train2, base_train1)

    mse_train1 = get_mse_1(dfngram1_train, p_table_train1, base_train1)
    mse_test1 = get_mse_1(dfngram1_test, p_table_train1, base_train1)

    mse_train = [mse_train1, mse_train2, mse_train3, mse_train4, mse_train5, mse_train6, mse_train7] 
    mse_test = [mse_test1, mse_test2, mse_test3, mse_test4, mse_test5, mse_test6, mse_test7]
    mse = [mse_train, mse_test]
    
    col_name = ['ngram_2','ngram_3','ngram_4','ngram_5', 'ngram_6','ngram_7','ngram_8']  
    mse_data = pd.DataFrame(data = mse)
    mse_data.columns = col_name
    mse_data.index = ['train_mse', 'test_mse']
    
    mse_data.to_csv(file_name, sep=',', encoding='utf-8')
    
    return mse_data


# # Get the best n-gram

# In[39]:


'''
dfngram1_train = pd.read_csv('../data/small_101_1.csv')
dfngram2_train = pd.read_csv('../data/small_101_2.csv')
dfngram3_train = pd.read_csv('../data/small_101_3.csv')
dfngram4_train = pd.read_csv('../data/small_101_4.csv')
dfngram5_train = pd.read_csv('../data/small_101_5.csv')
dfngram6_train = pd.read_csv('../data/small_101_6.csv')
dfngram7_train = pd.read_csv('../data/small_101_7.csv')

dfngram1_test = pd.read_csv('../data/small_101_1.csv')
dfngram2_test = pd.read_csv('../data/small_101_2.csv')
dfngram3_test = pd.read_csv('../data/small_101_3.csv')
dfngram4_test = pd.read_csv('../data/small_101_4.csv')
dfngram5_test = pd.read_csv('../data/small_101_5.csv')
dfngram6_test = pd.read_csv('../data/small_101_6.csv')
dfngram7_test = pd.read_csv('../data/small_101_7.csv')
'''


# In[66]:


'''
dfngram1_train = pd.read_csv('../data/ngram_1990_2009_1.csv')
dfngram2_train = pd.read_csv('../data/ngram_1990_2009_2.csv')
dfngram3_train = pd.read_csv('../data/ngram_1990_2009_3.csv')
dfngram4_train = pd.read_csv('../data/ngram_1990_2009_4.csv')
dfngram5_train = pd.read_csv('../data/ngram_1990_2009_5.csv')
dfngram6_train = pd.read_csv('../data/ngram_1990_2009_6.csv')
dfngram7_train = pd.read_csv('../data/ngram_1990_2009_7.csv')

dfngram1_test = pd.read_csv('../data/ngram_2010_2016_1.csv')
dfngram2_test = pd.read_csv('../data/ngram_2010_2016_2.csv')
dfngram3_test = pd.read_csv('../data/ngram_2010_2016_3.csv')
dfngram4_test = pd.read_csv('../data/ngram_2010_2016_4.csv')
dfngram5_test = pd.read_csv('../data/ngram_2010_2016_5.csv')
dfngram6_test = pd.read_csv('../data/ngram_2010_2016_6.csv')
dfngram7_test = pd.read_csv('../data/ngram_2010_2016_7.csv')
'''


# In[68]:

#fileName = "../data/mse_train_test_2010.csv"
mse_data = get_train_test(dfngram1_train, dfngram2_train, dfngram3_train, dfngram4_train, dfngram5_train,                   dfngram6_train, dfngram7_train, dfngram1_test, dfngram2_test, dfngram3_test,                   dfngram4_test, dfngram5_test, dfngram6_test, dfngram7_test, file_name = fileName)


# ### Get Graph

# In[58]:

fileName = "../data/mse_train_test_2010.csv"
df = pd.read_csv(fileName)
df = df.drop("Unnamed: 0", axis = 1)
df.index = ['train_mse', 'test_mse']
col_name = ["order = {}".format(i+1) for i in range(7)]
df.columns = col_name

ax1 = df.T.plot(lw=5, figsize=(10,6))
ax1.set_title("mse over order",fontsize=20)
ax1.set_xlabel("order",fontsize=18)
ax1.set_ylabel("mse",fontsize=18)

plt.savefig("../data/mse_train_test_2010.png")
#plt.savefig("../data/temp.png")

plt.show()


# # Combined with Twitter data  

# In[19]:

def separate_data(data):
    
    df = data[(data['next_rank'] < 101) & (data['next_rank'] > 0) & (data['current_rank'] < 101)]
    
    df0 = df[df['past_rank_1']>100]
    df1 = df[df['past_rank_1'] <=100]
    
    df1_only = df1[df1['past_rank_2']>100]
    df2 = df1[df1['past_rank_2'] <=100]
    
    df2_only = df2[df2['past_rank_3']>100]
    df3 = df2[df2['past_rank_3'] <=100]
    
    df3_only = df3[df3['past_rank_4']>100]
    df4 = df3[df3['past_rank_4'] <=100]
    
    return df, df0, df1, df1_only, df2, df2_only, df3, df3_only, df4 


# In[45]:

def data_twitter_reg(df, p_table2, p_table1, base2, base1, file_name = "../data/data_twitter_reg_temp"):
    
    df = df.drop("Unnamed: 0", axis = 1)
    df['past_rank_0'] = df['current_rank']
    
    ngram = 2
    col = ["past_rank_{}".format(i) for i in sorted(range(ngram), reverse=True)]
    df2 = df[col]
    
    diff_list = []
    est_list = []
    for i in range(len(df2)):
        basei = tuple(df2.iloc[i])
        esti = get_est_2(basei, p_table2, p_table1, base2, base1)
        diff_list.append(df['next_rank'].values[i] - esti)
        est_list.append(esti)
    
    df['diff'] = diff_list
    df['est'] = est_list
    
    df.to_csv(file_name, sep=',', encoding='utf-8')
    
    return df
    


# In[ ]:




# In[21]:

'''
dfngram1 = pd.read_csv('../data/small_101_1.csv')
dfngram2 = pd.read_csv('../data/small_101_2.csv')
'''


# In[59]:

'''
dfngram1 = pd.read_csv('../data/ngram101_1.csv')
dfngram2 = pd.read_csv('../data/ngram101_2.csv')
'''


# In[61]:

len(dfngram1), len(dfngram2)


# In[50]:

p_table1, base1 = get_markov(dfngram1, ngram = 1)
p_table2, base2 = get_markov(dfngram2, ngram = 2)


# In[23]:

df_data_csv = pd.read_csv('../data/result_database34_20160430.csv')


# In[24]:

df, df0, df1, df1_only, df2, df2_only, df3, df3_only, df4  = separate_data(df_data_csv)
print len(df_data_csv), len(df), len(df1), len(df2), len(df3)


# In[54]:

#fileName = "../data/data_twitter_reg_temp.csv"
fileName = "../data/data_twitter_reg.csv"
data_twitter = data_twitter_reg(df1, p_table2, p_table1, base2, base1, file_name = fileName)


# In[ ]:



