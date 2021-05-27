# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:2:21 2021

@author: e399410
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

# Open JSON file (MemoryError for user and review)
business_data = [json.loads(line) for line in open('yelp_academic_dataset_business.json', 'r',encoding="utf8")]
checkin_data = [json.loads(line) for line in open('yelp_academic_dataset_checkin.json', 'r',encoding="utf8")]
tip_data = [json.loads(line) for line in open('yelp_academic_dataset_tip.json', 'r',encoding="utf8")]
#user_data = [json.loads(line) for line in open('yelp_academic_dataset_user.json', 'r',encoding="utf8")]
#review_data = [json.loads(line) for line in open('yelp_academic_dataset_review.json', 'r',encoding="utf8")]

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
print(len(restaurants))
print(len(bars))
print(len(other))

# Dimensionality Reduction

# Data Mining Analysis