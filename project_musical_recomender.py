#!/usr/bin/env python
# coding: utf-8

# In this exercise, use NMF to recommender popular music artists! 
# 
# we have 2 scv files
# 
# 1. 100 artist name
# 
# 2. 24800 samples include 500 user
# 
# that merge those toghter and create sparse array artists rows whose rows correspond to artists and columns whose correspond to users. The entries give the number of times each artist was listened to by each user.**
# 
# In this exercise, build a pipeline and transform the array into normalized NMF features. The first step in the pipeline, MaxAbsScaler, transforms the data so that all users have the same influence on the model, regardless of how many different artists they've listened to.
# 
# then we use the resulting normalized NMF features for recommendation with NMF and cosine similarities!

# Perform the necessary imports
# 

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline


df = pd.read_csv('scrobbler-small-sample.csv')


artists_df=pd.read_csv('artists.csv', header=None, names=['artist_name'])
artists_df.head()

artist_names=artists_df['artist_name']


# then add artist_offset column to artists_df to have same column that merge with df

artists_df['artist_offset'] = artists_df.index


# then, we merge the 2 data frames obtained from csv files together. To have a single data frame
temp_df=pd.merge(df,artists_df,on="artist_offset")


temp_df.head()


artists=temp_df.pivot_table(columns = 'user_offset',index = 'artist_name',values='playcount',fill_value=0)
artists.shape

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()


# Create an NMF model: nmf
nmf = NMF(n_components=20)


# Create a Normalizer: normalizer
normalizer = Normalizer()


# Create a pipeline: pipeline
pipeline = make_pipeline(scaler ,nmf ,normalizer)


# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)


# Create a DataFrame: df
NMF_df = pd.DataFrame(norm_features,index=artist_names)
NMF_df.shape


# Select row of 'Bruce Springsteen': artist
artist = NMF_df.loc['Bruce Springsteen']


# Compute cosine similarities: similarities
similarities =NMF_df.dot(artist)
#NMF_df(111,20) matrix dot artist(20,1) matrix equal similarities(500,1) matrix

# Display those with highest cosine similarity
print(similarities.nlargest())