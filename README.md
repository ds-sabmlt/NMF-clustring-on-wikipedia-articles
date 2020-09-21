# saber.mellati

In this exercise, use NMF to recommender popular music artists!

we have 2 scv files

100 artist name

24800 samples include 500 user

that merge those toghter and create sparse array artists rows whose rows correspond to artists and columns whose correspond to users. The entries give the number of times each artist was listened to by each user.**

In this exercise, build a pipeline and transform the array into normalized NMF features. The first step in the pipeline, MaxAbsScaler, transforms the data so that all users have the same influence on the model, regardless of how many different artists they've listened to.

then we use the resulting normalized NMF features for recommendation with NMF and cosine similarities!
