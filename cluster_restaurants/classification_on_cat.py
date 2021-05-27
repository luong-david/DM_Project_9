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
#For random forest stuff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics
from sklearn.tree import export_graphviz

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
all_category_names = vectorizer.get_feature_names()



#vectorized data is esesentially the input, the categories of the data
#all_ratings are the labels
X_train, X_test, y_train, y_test = train_test_split(vectorized_matrix, all_ratings, test_size=0.2)

#round the labels (test data) so we have strict categories determined by the 
#5 star ratings. Also turn into 1d array as that's what classifier expects
y_train = np.round(y_train).ravel()
y_test = np.round(y_test).ravel()

clf=RandomForestClassifier(n_estimators=100)

start_time = time.time()
clf.fit(X_train,y_train)
print('Fitting took ', time.time() - start_time, ' seconds')
start_time = time.time()
y_pred=clf.predict(X_test)

print('Predicting took ', time.time() - start_time, ' seconds')
print("Accuracy of prediction being:",metrics.accuracy_score(y_test, y_pred))
feature_imp = pd.Series(clf.feature_importances_,index=all_category_names).sort_values(ascending=False)

sns.barplot(x=feature_imp[0:25], y=feature_imp.index[0:25])
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

#Let's see if classification accuracy improves by using the top 100 most
#important features only
tmp = feature_imp.index
category_names_ordered = list(tmp)
important_feature_count = 25
reduced_feature_set = csr_matrix((vectorized_matrix.shape[0],important_feature_count), dtype=np.int64)
reduced_ratings = [] #<- don't use. Ratings won't change for each restaurant
reduced_category_names = []
#resort but this time put the worst features at the top
#feature_imp = pd.Series(clf.feature_importances_,index=all_category_names).sort_values()

for i in range(0,important_feature_count):
    cat = category_names_ordered[i]
    #find corresponding feature index in the vectorized matrix
    cat_index = vectorizer.vocabulary_[cat]
    #assign into reduced matrix, note that the ordering is changing.
    #Should come back and verify this all makes sense
    reduced_feature_set[:,i] = vectorized_matrix[:, cat_index] 
    #one easy check is to make this occur from back to front, instead of front to back.
    #Then we'd know that the order isn't affecting anything
    reduced_category_names.append(cat)

#Now repeat:
X_train, X_test, y_train, y_test = train_test_split(reduced_feature_set, all_ratings, test_size=0.2)

y_train = np.round(y_train).ravel()
y_test = np.round(y_test).ravel()

clf=RandomForestClassifier(n_estimators=100)
start_time = time.time()
clf.fit(X_train,y_train)
print('Fitting took ', time.time() - start_time, ' seconds')
start_time = time.time()
y_pred=clf.predict(X_test)

print('Predicting took ', time.time() - start_time, ' seconds')
print("Accuracy of prediction being:",metrics.accuracy_score(y_test, y_pred))
feature_imp = pd.Series(clf.feature_importances_,index=reduced_category_names).sort_values(ascending=False)
sns.barplot(x=feature_imp[0:25], y=feature_imp.index[0:25])
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


'''
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


'''