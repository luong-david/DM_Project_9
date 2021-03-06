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

def function1(tip_data, DR):
    
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
    
    #perform dimensionality reduction (memory issue if tips > 1000)
    if DR == 0:
        print('No DR performed')
        mat = mat3 
    elif DR == 1:
        print('Using TSNE for DR for 1000 data points due to memory issue')
        mat, tsne_result = func.DR_TSNE(mat3[0:1000])
        result = tsne_result
    elif DR == 2:
        print('Using PCA for DR for 1000 data points due to memory issue')
        mat, pca_result = func.DR_PCA(mat3[0:1000])
        result = pca_result
    else:
        print('Invalid DR method, proceed with no DR')
        mat = mat3   
    
    Ktarget = 10
    CLUSTERIND, INERTIAS = func.BSKM(mat,mat.shape[0],Ktarget)

    # assign cluster number to data vector
    k = 0
    cluster_solution = np.zeros(mat.shape[0],dtype=int)
    for C in CLUSTERIND:
        for item in C:
            cluster_solution[item-1] = k
        k += 1

    # Plot Total SSE vs K
    plt.figure()
    plt.plot(range(1,Ktarget+1),INERTIAS)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average SSE')
    
    # plot clusters
    if DR == 1 or DR == 2:
        func.plotClusters(result,cluster_solution)
    
    for i in range(Ktarget):
        print('===============================')
        print("Sample Reviews from Cluster", i+1)
        for j in random.sample(CLUSTERIND[i],min(5,len(CLUSTERIND[i]))):
            print(tip_data[j]['text'])
        print('===============================')