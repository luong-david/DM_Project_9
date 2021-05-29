# -*- coding: utf-8 -*-
"""
Created on Fri May 28 22:30:47 2021

@author: e399410
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

import functions as func

def function1(tip_data):
    
#    corpus = list(set([word for tip in tip_data for word in tip['text'].lower().split()]))
    corpus = set()
    for tip in tip_data:
        corpus.add(tip['text'])
        
    
    vectorizer = CountVectorizer()
    vectorizer.fit(corpus)
    mat1 = vectorizer.transform(corpus)
    all_words = vectorizer.get_feature_names()
    
    mat2 = func.csr_idf(mat1, copy=True)
    mat3 = func.csr_l2normalize(mat2, copy=True)
    
    Ktarget = 10
    CLUSTERIND, INERTIAS = func.BSKM(mat3,mat3.shape[0],Ktarget)
    
    # Plot Total SSE vs K
    plt.plot(range(1,Ktarget+1),INERTIAS)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average SSE')
    
    for i in range(Ktarget):
        print('===============================')
        print("Sample Reviews from Cluster", i+1)
        for j in random.sample(CLUSTERIND[i],min(5,len(CLUSTERIND[i]))):
            print(tip_data[j]['text'])
        print('===============================')