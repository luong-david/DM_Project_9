# -*- coding: utf-8 -*-
"""

Main driver file for data mining studies on Yelp Dataset

Created on Thu May 22 14:2:21 2021

@author: Thomas Jacob, David Luong, Taylor Maurer
"""

import json
import numpy as np
import scipy as sp
#matplotlib inline
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random
import bisect
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, eigs
from numpy.linalg import matrix_rank
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import functions as func
import task1 as tsk1
import task2 as tsk2

business = 1
checkin = 0
tip = 0
user = 0
review = 0
 
# Open JSON file (MemoryError for user and review)
if business:
    business_data = [json.loads(line) for line in open('yelp_dataset/yelp_academic_dataset_business.json', 'r',encoding="utf8")]
if checkin:
    checkin_data = [json.loads(line) for line in open('yelp_dataset/yelp_academic_dataset_checkin.json', 'r',encoding="utf8")]
if tip:
    tip_data = [json.loads(line) for line in open('yelp_dataset/yelp_academic_dataset_tip.json', 'r',encoding="utf8")]
if user:
    user_data = [json.loads(line) for line in open('yelp_academic_dataset_user.json', 'r',encoding="utf8")]
if review:
    review_data = [json.loads(line) for line in open('yelp_academic_dataset_review.json', 'r',encoding="utf8")] 

if business:
    # Split into restaurants, bars and things
    restaurants = []
    bars = []
    other = []
    for raw_dict in business_data:
    	appended = False
    	if raw_dict['categories'] != None:
    		if 'restaurants' in raw_dict['categories'] or 'Restaurants' in raw_dict['categories']:
    			restaurants.append(raw_dict)
    			appended = True
    		if 'bars' in raw_dict['categories'] or 'Bars' in raw_dict['categories']:
    			bars.append(raw_dict)
    			appended = True
    		if not appended:
    			other.append(raw_dict)
    print('Total number of restaurants in dataset: ', len(restaurants))
    print('Total number of bars in dataset: ', len(bars))
    print('Total number of other businesses in dataset: ', len(other))
    
if tip:
    tips = []
    # filter out tips that are less than 8 words long
    for raw_dict in tip_data:
        if len(raw_dict['text'].split()) > 8:
            tips.append(raw_dict)
    print('Total number of tips in dataset: ', len(tips))

# Data Mining Studies
if business:
    nR = 10000 #number of restaurants to data mine
    #tsk1.function1(restaurants[0:nR],bars,other)
    tsk1.function2(restaurants[0:nR],bars,other)
if tip:
    nT = 10000 #number of tips to data mine
    DR = [1] # 0 = no DR, 1 = TNSE 2 = PCA
    for dr in DR:
       tsk2.function1(tips[0:nT], dr)