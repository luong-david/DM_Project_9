


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
import seaborn as sns
import collections

def function0(restaurants, cluster_size_range):
    print('='*50)
    print('='*50)
    print('Running cluster analysis on business data')
    cluster_variance = []
    for c in cluster_size_range:
        print('='*50)
        print('='*50)
        print('Clustering with size ', c)
        v = function1(restaurants[0:-1], c)
        cluster_variance.append(v)
    plt.scatter(cluster_size_range, cluster_variance)
    plt.xlabel('Cluster Number (k)')
    plt.ylabel('Variance of Ratings per Cluster')
    plt.title('Rating Variance With Increasing K')
    plt.show()

#Taking the David Luong approach
def function1(restaurants, cluster_number, \
    cluster_statistics_size = 5, top_word_size = 10):
    print('Running clustering approach for task 1')
    all_cats = []
    all_ratings = np.zeros((len(restaurants),1))
    restaurant_ind = 0
    #Run through all restaurants:
    for restaurant in restaurants:
        #extract the categories
        cats = restaurant['categories']
        all_cats.append(cats)
        #extract star rating
        all_ratings[restaurant_ind, 0] = restaurant['stars']
        restaurant_ind+=1
    
    #Vectorize the categories:
    vectorizer = CountVectorizer()
    vectorizer.fit(all_cats)
    vectorized_matrix = vectorizer.transform(all_cats)

    #time the Kmeans fit/clustering process
    print('='*50)
    print('Running KMeans Clustering')
    start_time = time.time()
    kmeans = KMeans(n_clusters=cluster_number, random_state = 0). fit(vectorized_matrix)
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
    cluster_sizes = []
    #Now go through and aggregate the ratings, and print out statistics
    #associated wiht the cluster
    for cluster_label in cluster_ratings:
	    avg_rating.append(np.mean(cluster_ratings[cluster_label]))
	    assc_cluster.append(cluster_label)
	    cluster_sizes.append(len(cluster_ratings[cluster_label]))


    print('Average rating of all restaurants is ', np.mean(all_ratings), ' stars')

    function2(assc_cluster, avg_rating)
    function3(assc_cluster, cluster_sizes)
    print('='*50)
    print('Analyzing cluster data')
    avg_rating = np.array(avg_rating)
    assc_cluster = np.array(assc_cluster)
    #Get the sorted indices, noting worst are at the front
    sorted_indices = np.argsort(avg_rating)
    #Now go through sorted indices and get out the top five rated
    #clusters, and then get their assocaited words
    key_list = list(vectorizer.vocabulary_.keys())
    val_list = list(vectorizer.vocabulary_.values())
    print('='*50)
    print('Getting top words associated with worst rated clusters')
    for i in sorted_indices[0:cluster_statistics_size]:
	    cluster_label = assc_cluster[i]
	    cluster_center = kmeans.cluster_centers_[cluster_label, :]
	    print('-'*50)
	    print('Cluster ', cluster_label)
	    print('Avg rating of ', avg_rating[i])
	    print('Most common words used were')
	    #Sort the cluster center so that the most frequent words 
	    #are at the back. Then flip so they're at the front
	    cc_sorted_indices= np.flip(np.argsort(cluster_center))
	    for j in cc_sorted_indices[0:top_word_size]:
		    #so j is the index within the feature vector
		    #that corresponds to the most frequently used words for
		    #this cluster
		    #Get the value this index corresponds to in feature vector
		    frequent_index = val_list.index(j)
		    #Get the word that is associated with
		    frequency_word = key_list[frequent_index]
		    print(frequency_word, end=' ')
	    print('')
    print('='*50)
    print('Getting top words associated with best rated clusters')
    #flip s.t. the best rated clusters are at the front
    sorted_indices = np.flip(np.argsort(avg_rating))
    #Now go through sorted indices and get out the top five rated
    #clusters, and then get their assocaited words
    key_list = list(vectorizer.vocabulary_.keys())
    val_list = list(vectorizer.vocabulary_.values())
    for i in sorted_indices[0:cluster_statistics_size]:
	    cluster_label = assc_cluster[i]
	    cluster_center = kmeans.cluster_centers_[cluster_label, :]
	    print('-'*50)
	    print('Cluster ', cluster_label)
	    print('Avg rating of ', avg_rating[i])
	    print('Most common words used were')
	    #Sort the cluster center so that the most frequent words 
	    #are at the back. Then flip so they're at the front
	    cc_sorted_indices= np.flip(np.argsort(cluster_center))
	    for j in cc_sorted_indices[0:top_word_size]:
		    #so j is the index within the feature vector
		    #that corresponds to the most frequently used words for
		    #this cluster
		    #Get the value this index corresponds to in feature vector
		    frequent_index = val_list.index(j)
		    #Get the word that is associated with
		    frequency_word = key_list[frequent_index]
		    print(frequency_word, end=' ')
	    print('')
    print('='*50)
    print('Getting top words associated with most average rated clusters')
    #flip s.t. the best rated clusters are at the front
    sorted_indices = np.argsort(avg_rating)
    #Now go through sorted indices and get out the top five rated
    #clusters, and then get their assocaited words
    key_list = list(vectorizer.vocabulary_.keys())
    val_list = list(vectorizer.vocabulary_.values())
    start_index = int(np.floor(len(sorted_indices)/2)) #<- halfway point to get avg clusters
    for i in sorted_indices[start_index:(start_index + cluster_statistics_size)]:
	    cluster_label = assc_cluster[i]
	    cluster_center = kmeans.cluster_centers_[cluster_label, :]
	    print('-'*50)
	    print('Cluster ', cluster_label)
	    print('Avg rating of ', avg_rating[i])
	    print('Most common words used were')
	    #Sort the cluster center so that the most frequent words 
	    #are at the back. Then flip so they're at the front
	    cc_sorted_indices= np.flip(np.argsort(cluster_center))
	    for j in cc_sorted_indices[0:top_word_size]:
		    #so j is the index within the feature vector
		    #that corresponds to the most frequently used words for
		    #this cluster
		    #Get the value this index corresponds to in feature vector
		    frequent_index = val_list.index(j)
		    #Get the word that is associated with
		    frequency_word = key_list[frequent_index]
		    print(frequency_word, end=' ')
	    print('')


    #bad_words = ['fast', 'wings', 'burgers', 'traditional']
    bad_words = ['fast', 'wings', 'traditional']
    #bad_words = ['fast']
    bad_word_index = []
    #good_words = ['trucks', 'mediterranean', 'fusion', 'specialty']
    good_words = ['trucks', 'new', 'specialty']#, 'seafood', 'nightlife']
    #good_words = ['trucks']
    good_word_index = []
    #first convert bad_words/good_words to indices associated with the vector:
    for bad_word in bad_words:
        bad_word_index.append(vectorizer.vocabulary_[bad_word])
    for good_word in good_words:
        good_word_index.append(vectorizer.vocabulary_[good_word])




    function4(assc_cluster, avg_rating, kmeans.cluster_centers_, bad_word_index, good_word_index)
    
    return np.var(avg_rating)
def function4(unsorted_clusters, avg_rating, cluster_centers, bad_word_index, good_word_index):
    #From previous analysis the given categories of truck, indicate goodness and category of fast indicate 'badness'
    #for scalabnility let's allow for >1 word to be entered as well to create this plot
    sorted_indices = np.argsort(unsorted_clusters)
    sorted_clusters = np.take_along_axis( \
        np.array(unsorted_clusters), sorted_indices, axis = 0)
    sorted_ratings = np.take_along_axis( \
        np.array(avg_rating), sorted_indices, axis = 0)
    badness_measures = []
    goodness_measures = []
    for i in range(0,len(sorted_clusters)):
        c = sorted_clusters[i]
        cluster_center = cluster_centers[c, :]
        badness_measures.append(0)
        goodness_measures.append(0)
        
        #get the number of bad words associated with this cluster center
        for bad_word_i in bad_word_index:
            badness_measures[i] += cluster_center[bad_word_i]
        for good_word_i in good_word_index:
            goodness_measures[i] += cluster_center[good_word_i]
        #At the end of each loop you'll have a goodness_measure and a badness_measure for each cluster
        #based on the number of times that the word appeared
    
    #Basically goodness = green, badness = red
    max_goodness = np.max(goodness_measures)
    max_badness = np.max(badness_measures)


    fig = plt.figure()
    for i in range(0, len(sorted_clusters)):
        #compute red:
        rval = badness_measures[i]/max_badness
        gval = goodness_measures[i]/max_goodness
        plt.scatter(sorted_clusters[i], sorted_ratings[i], color=[rval, gval, 0])
    plt.title('Clusters vs. Cluster Rating (K=100)')
    plt.xlabel('Cluster ID')
    plt.ylabel('Star Rating')
    plt.show()






def function2(unsorted_clusters, associated_ratings):
    print('='*50)
    print('Plotting cluster results')
    
    #sort cluster list and the associated ratings as well:
    sorted_indices = np.argsort(unsorted_clusters)
    sorted_clusters = np.take_along_axis( \
        np.array(unsorted_clusters), sorted_indices, axis = 0)
    sorted_ratings = np.take_along_axis( \
        np.array(associated_ratings), sorted_indices, axis = 0)
    sns.scatterplot(sorted_clusters, sorted_ratings)
    plt.title('Clusters vs. Cluster Average Rating')
    plt.xlabel('cluster number')
    plt.ylabel('Cluster Average Rating')
    plt.show()
def function3(unsorted_clusters, associated_size):
    print('='*50)
    print('Plotting cluster results')
    
    #sort cluster list and the associated ratings as well:
    sorted_indices = np.argsort(unsorted_clusters)
    sorted_clusters = np.take_along_axis( \
        np.array(unsorted_clusters), sorted_indices, axis = 0)
    sorted_ratings = np.take_along_axis( \
        np.array(associated_size), sorted_indices, axis = 0)
    sns.scatterplot(sorted_clusters, sorted_ratings)
    plt.title('Clusters vs. Cluster Size')
    plt.xlabel('Cluster Number')
    plt.ylabel('Cluster Size')
    plt.show()

    
