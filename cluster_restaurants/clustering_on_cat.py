# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:2:21 2021

@author: e399410
"""
import time
import json
import re
import numpy as np
import scipy as sp
#matplotlib inline
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import random
import bisect
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, eigs
from numpy.linalg import matrix_rank
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Open JSON file (MemoryError for user and review)
business_data = [json.loads(line) for line in open('yelp_dataset/yelp_academic_dataset_business.json', 'r',encoding="utf8")]
#checkin_data = [json.loads(line) for line in open('yelp_academic_dataset_checkin.json', 'r',encoding="utf8")]
#tip_data = [json.loads(line) for line in open('yelp_academic_dataset_tip.json', 'r',encoding="utf8")]
#user_data = [json.loads(line) for line in open('yelp_academic_dataset_user.json', 'r',encoding="utf8")]
#review_data = [json.loads(line) for line in open('yelp_academic_dataset_review.json', 'r',encoding="utf8")]

# Split into restaurants, bars and things
restaurants = []
bars = []
other = []
for raw_dict in business_data:
	appended = False
	if raw_dict['categories'] != None:
		if 'restaurant' in raw_dict['categories'] or 'Restaurant' in raw_dict['categories']:
			restaurants.append(raw_dict)
			appended = True
		if 'bars' in raw_dict['categories'] or 'Bars' in raw_dict['categories']:
			bars.append(raw_dict)
			appended = True
		if not appended:
			other.append(raw_dict)
print(len(restaurants))
print(len(bars))
print(len(other))


#now go through all categories of the restaurants
'''
cat_dictionary = set()
for restaurant in restaurants:
    #get list of categories
    cats = restaurant['categories']
    #split via comma
    split_string = re.split(', |,| ', cats)
    for word in split_string:
        #remove special characters
        clean_string = re.sub('\W+', '', word)
        cat_dictionary.add(clean_string.lower())


cat_dictionary = sorted(cat_dictionary)
'''
all_cats = []
all_ratings = np.zeros((len(restaurants),1))
restaurant_ind = 0
for restaurant in restaurants:
    #get categories
	cats = restaurant['categories']
	all_cats.append(cats)
	all_ratings[restaurant_ind, 0] = restaurant['stars']
	restaurant_ind+=1
		
    

vectorizer = CountVectorizer()
vectorizer.fit(all_cats)
vectorized_matrix = vectorizer.transform(all_cats)

start_time = time.time()
#Use n_clusters for 5 to ideally seperate into 5 ratings...
kmeans = KMeans(n_clusters=50, random_state=0).fit(vectorized_matrix)
print('Clustering took ', time.time() - start_time, ' seconds')
cluster_ratings = {}
cluster_cats = {}
restaurant_ind = 0
for label in kmeans.labels_:
	#within kmeans.labels_ each restaurant has been given a label
	#Go through, find the star rating associated ith this label and append it
	#to the cluster_rating dictionary for that cluster
	if label in cluster_ratings:
		cluster_ratings[label].append(all_ratings[restaurant_ind,0])
		cluster_cats[label].append(all_cats[restaurant_ind]) #<- inefficient, but ok
	
	else:
		cluster_ratings[label] = []
		cluster_cats[label] = []
		cluster_ratings[label].append(all_ratings[restaurant_ind,0])
		cluster_cats[label].append(all_cats[restaurant_ind])
	restaurant_ind += 1

avg_rating = []
assc_cluster = []
for cluster_label in cluster_ratings:
	avg_rating.append(np.mean(cluster_ratings[cluster_label]))
	assc_cluster.append(cluster_label)
	print('Cluster ', cluster_label, ' statistics:')
	print('-Average rating is ', avg_rating[-1], ' stars')
	print('-Size is ', len(cluster_ratings[cluster_label]))
print('Average rating of all restaurants is ', np.mean(all_ratings), ' stars')


avg_rating = np.array(avg_rating)
assc_cluster = np.array(assc_cluster)
#Get the sorted indices, noting worst are at the front
sorted_indices = np.argsort(avg_rating)
#Now go through sorted indices and get out the top five rated
#clusters, and then get their assocaited words
key_list = list(vectorizer.vocabulary_.keys())
val_list = list(vectorizer.vocabulary_.values())
for i in sorted_indices[0:5]:
	cluster_label = assc_cluster[i]
	cluster_center = kmeans.cluster_centers_[cluster_label, :]
	print('-'*50)
	print('Cluster ', cluster_label)
	print('Avg rating of ', avg_rating[i])
	print('Most common words used were')
	#Sort the cluster center so that the most frequent words 
	#are at the back. Then flip so they're at the front
	cc_sorted_indices= np.flip(np.argsort(cluster_center))
	for j in cc_sorted_indices[0:10]:
		#so j is the index within the feature vector
		#that corresponds to the most frequently used words for
		#this cluster
		#Get the value this index corresponds to in feature vector
		frequent_index = val_list.index(j)
		#Get the word that is associated with
		frequency_word = key_list[frequent_index]
		print(frequency_word, end=' ')
	print('')


