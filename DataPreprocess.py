
# coding: utf-8

# In[1]:


import numpy as np
import numpy.linalg as la
import scipy.io
import pandas as pd


# In[2]:


# open .csv
genuine_users_raw = pd.read_csv('./data/users_genuine.csv', sep=',')
spambot_users_raw = pd.read_csv('./data/users_spambot.csv', sep=',')

genuine_tweets_raw = pd.read_csv('./data/tweets_genuine.csv', sep=',')
spambot_tweets_raw = pd.read_csv('./data/tweets_spambot.csv', sep=',')


# In[3]:


# select features
genuine_users = genuine_users_raw[['id',
               'statuses_count',
               'followers_count',
               'friends_count',
               'favourites_count',
               'listed_count',
               'verified',
               'geo_enabled',
               'timestamp']]

spambot_users = spambot_users_raw[['id',
               'statuses_count',
               'followers_count',
               'friends_count',
               'favourites_count',
               'listed_count',
               'verified',
               'geo_enabled',
               'timestamp']]

genuine_tweets = genuine_tweets_raw[['user_id',
                                     'favorite_count',
                                     'retweet_count',
                                     'num_hashtags',
                                     'num_urls',
                                     'num_mentions']]

spambot_tweets = spambot_tweets_raw[['user_id',
                                     'favorite_count',
                                     'retweet_count',
                                     'num_hashtags',
                                     'num_urls',
                                     'num_mentions']]


# In[4]:


# data preprocessing


# In[5]:


# remove NaN
genuine_users = genuine_users.fillna(0)
spambot_users = spambot_users.fillna(0)
genuine_tweets = genuine_tweets.fillna(0)
spambot_tweets = spambot_tweets.fillna(0)


# In[6]:


# convert account created_at to account age (months), add acct_age column, remove created_at column

age = (pd.to_datetime('2018-11-09 00:00:00') - pd.to_datetime(genuine_users['timestamp']))/np.timedelta64(1, 'M')
genuine_users.loc[:,'acct_age'] = age.values
genuine_users = genuine_users.drop('timestamp',1)

age = (pd.to_datetime('2018-11-09 00:00:00') - pd.to_datetime(spambot_users['timestamp']))/np.timedelta64(1, 'M')
spambot_users.loc[:,'acct_age'] = age.values
spambot_users = spambot_users.drop('timestamp',1)


# In[7]:


# aggregate tweet data by user id
# for genuine users
new_features = ['favorite_count','retweet_count','num_hashtags', 'num_urls','num_mentions']

# create empty columns to fill in user df
for f in new_features:
    genuine_users[f] = np.nan

# get unique user ids from tweet df
unique_ids = genuine_tweets['user_id'].unique()
indices = []
for unique_id in unique_ids:
    index = genuine_users.loc[genuine_users['id']==unique_id].index[0]
    indices.append(index)

    mean = genuine_tweets.loc[genuine_tweets['user_id'] == unique_id]['favorite_count'].mean()
    genuine_users.iloc[index, genuine_users.columns.get_loc('favorite_count')] = mean

    mean = genuine_tweets.loc[genuine_tweets['user_id'] == unique_id]['retweet_count'].mean()
    genuine_users.iloc[index, genuine_users.columns.get_loc('retweet_count')] = mean

    mean = genuine_tweets.loc[genuine_tweets['user_id'] == unique_id]['num_hashtags'].mean()
    genuine_users.iloc[index, genuine_users.columns.get_loc('num_hashtags')] = mean

    mean = genuine_tweets.loc[genuine_tweets['user_id'] == unique_id]['num_urls'].mean()
    genuine_users.iloc[index, genuine_users.columns.get_loc('num_urls')] = mean

    mean = genuine_tweets.loc[genuine_tweets['user_id'] == unique_id]['num_mentions'].mean()
    genuine_users.iloc[index, genuine_users.columns.get_loc('num_mentions')] = mean


# for spambot users
# create empty columns to fill in user df
for f in new_features:
    spambot_users[f] = np.nan

indices2 = []
# get unique user ids from tweet df
unique_ids = spambot_tweets['user_id'].unique()
for unique_id in unique_ids:
    index = spambot_users.loc[spambot_users['id']==unique_id].index[0]
    indices2.append(index)

    mean = spambot_tweets.loc[spambot_tweets['user_id'] == unique_id]['favorite_count'].mean()
    spambot_users.iloc[index, spambot_users.columns.get_loc('favorite_count')] = mean

    mean = spambot_tweets.loc[spambot_tweets['user_id'] == unique_id]['retweet_count'].mean()
    spambot_users.iloc[index, spambot_users.columns.get_loc('retweet_count')] = mean

    mean = spambot_tweets.loc[spambot_tweets['user_id'] == unique_id]['num_hashtags'].mean()
    spambot_users.iloc[index, spambot_users.columns.get_loc('num_hashtags')] = mean

    mean = spambot_tweets.loc[spambot_tweets['user_id'] == unique_id]['num_urls'].mean()
    spambot_users.iloc[index, spambot_users.columns.get_loc('num_urls')] = mean

    mean = spambot_tweets.loc[spambot_tweets['user_id'] == unique_id]['num_mentions'].mean()
    spambot_users.iloc[index, spambot_users.columns.get_loc('num_mentions')] = mean


# In[8]:


# drop users with no tweet data
genuine_users = genuine_users.dropna(how='any')
spambot_users = spambot_users.dropna(how='any')


# In[9]:


#  normalize values over feauture to range 0,1
features = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
                'verified', 'geo_enabled', 'acct_age', 'favorite_count', 'retweet_count', 'num_hashtags', 'num_urls',
            'num_mentions']

for feature in features:
    max_value = np.maximum(genuine_users[feature].max(), spambot_users[feature].max())
    genuine_users[feature] = genuine_users[feature]/max_value
    spambot_users[feature] = spambot_users[feature]/max_value


# In[10]:


genuine_users['bot_or_not'] = 0
spambot_users['bot_or_not'] = 1


# In[11]:


dataframes = [spambot_users, genuine_users]
all_data = pd.concat(dataframes)


# In[22]:


# write to csv
all_data.to_csv('project_data.csv', sep=',')


# In[13]:


genuine_users.head()
