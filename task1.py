# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:26:52 2021

@author: David Luong
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def function1(restaurants,bars,other):
    print('Doing function1 in task1')
    
    #go through different labels
    labels = ['postal_code','review_count','stars', 'city', 'state']
    for lab in labels:
    
        #now go through all categories of the restaurants
    
        all_cats = []
        all_labels = []
        restaurant_ind = 0
        for restaurant in restaurants:
            #get categories
        	cats = restaurant['categories']
        	all_cats.append(cats)
        	all_labels.append(restaurant[lab])
        	restaurant_ind+=1
        
        vectorizer = CountVectorizer()
        
        vectorizer.fit(all_cats)
        vectorized_matrix = vectorizer.transform(all_cats)
        all_category_names = vectorizer.get_feature_names()
        
        #vectorized data is esesentially the input, the categories of the data
        #all_ratings are the labels
        X_train, X_test, y_train, y_test = train_test_split(vectorized_matrix, all_labels, test_size=0.2)
        
        if lab == 'stars':
            #round the labels (test data) so we have strict categories determined by the 
            #5 star ratings. Also turn into 1d array as that's what classifier expects
            y_train = np.round(y_train).ravel()
            y_test = np.round(y_test).ravel()
    
        clf = RandomForestClassifier(max_depth=20, random_state=0)
        clf.fit(X_train, y_train)
        print('==================================')
        print(lab + " accuracy is", clf.score(X_train,y_train))
        
        y_pred = clf.predict(X_test)
        print('The '+ lab + ' test F1-Score is ', f1_score(y_test,y_pred, average = 'micro'))
        print('==================================')