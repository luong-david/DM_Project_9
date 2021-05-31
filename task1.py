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
import math

import functions as func

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
            if lab == 'review_count':
                all_labels.append(str(np.round(np.array(restaurant[lab]),-3)))
            else:
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
            # convert labels of floats to ints to strings
            all_string_labels = []
            for item in all_labels:
                all_string_labels.append(str(int(item)))
            all_labels = all_string_labels
            
        if lab == 'city':
            # convert labels to lower case and remove whitespaces
            all_string_labels = []
            for item in all_labels:
                all_string_labels.append(item.lower().strip())
            all_labels = all_string_labels
    
        clf = RandomForestClassifier(max_depth=20, random_state=0)
        clf.fit(X_train, y_train)
        print('==================================')
        print(lab + " accuracy is", clf.score(X_train,y_train))
        
        y_pred = clf.predict(X_test)
        print('The '+ lab + ' test F1-Score is ', f1_score(y_test,y_pred, average = 'micro'))
        print('Number of Restaurants: ', restaurant_ind)
        print('==================================')
        
        # Visualize a Decision Tree
        func.plotDecisionTree(clf,all_category_names,all_labels,'classifier_'+lab)

def function2(restaurants,bars,other):
    print('Doing function2 in task1')
    
    # get list of all attributes
    labels = func.getAttributesList(restaurants)
    
    #go through different labels
    for lab in labels:
    
        #now go through all categories of the restaurants
    
        all_cats = []
        all_labels = []
        restaurant_ind = 0
        for restaurant in restaurants:
            if restaurant['attributes'] is not None:
                if func.checkKey(restaurant['attributes'],lab):
                    #get categories
                   	cats = restaurant['categories']
                   	all_cats.append(cats)
                   	all_labels.append(restaurant['attributes'][lab])
                   	restaurant_ind+=1

        # skip the attribute if not enough restaurants have it ( < 10)
        if restaurant_ind < 10:
            continue
        
        vectorizer = CountVectorizer()
        vectorizer.fit(all_cats)
        vectorized_matrix = vectorizer.transform(all_cats)
        all_category_names = vectorizer.get_feature_names()
        
        #vectorized data is esesentially the input, the categories of the data
        #all_ratings are the labels
        X_train, X_test, y_train, y_test = train_test_split(vectorized_matrix, all_labels, test_size=0.2)
            
        clf = RandomForestClassifier(max_depth=20, random_state=0)
        clf.fit(X_train, y_train)
        print('==================================')
        print(lab + " accuracy is", clf.score(X_train,y_train))
        
        y_pred = clf.predict(X_test)
        print('The '+ lab + ' test F1-Score is ', f1_score(y_test,y_pred, average = 'micro'))
        print('Number of Restaurants: ', restaurant_ind)
        print('==================================')
        
        # Visualize a Decision Tree
        func.plotDecisionTree(clf,all_category_names,all_labels,'classifier_'+lab)