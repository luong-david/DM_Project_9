# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:29:50 2021

@author: e399410
"""

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix

def SSE(mat,d1,centroid):
    diff = mat[d1,:].toarray()-centroid
    sumdiffsquared = np.power(diff,2).sum()
    return sumdiffsquared

def checkKey(dict, key):
      
    if key in dict:
        return True
    else:
        return False
    
def getAttributesList(dict):
    att_set = set()
    for item in dict:
        if item['attributes'] is not None:
            for att in list(item['attributes'].keys()):
                att_set.add(att)
    return list(att_set)    

# scale matrix and normalize its rows
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
    
    if copy is True:
        return mat
    
def BSKM(mat,nrows,Ktarget):

    clusterind = list(range(0,nrows))
    CLUSTERIND = []
    CENTROIDS = []
    INERTIAS = []
    SSES= []
    K = 0
    
    # Initialize the list of clusters to contain the cluster containing all points
    cluster_selected = clusterind
    
    while K < Ktarget:
               
        # Select a cluster
        X = mat[cluster_selected]
        
        # Bisect selected cluster using K-means
        kmeans = KMeans(n_clusters=2, random_state=random.randint(0,1000)).fit(X)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        inertia = kmeans.inertia_

        # Add the two clusters from the bisection with the lowest SSE to the list of clusters
        count = 0
        count0 = 0
        count1 = 0
        sses = [0,0]
        clusterindA = []
        clusterindB = []
        for lab in labels:
            sses[lab] += SSE(mat,cluster_selected[count],centroids[lab])
            if lab == 0:
                clusterindA.append(cluster_selected[count])
                count0 += 1
            else:
                clusterindB.append(cluster_selected[count])
                count1 += 1
            count += 1
        # update clusters
        K += 1
        # append bisected clusters to list of clusters
        CLUSTERIND.append(clusterindA)
        CLUSTERIND.append(clusterindB)
        SSES.append(sses[0])
        SSES.append(sses[1])
        INERTIAS.append(sum(SSES)/K)
        
        if K == Ktarget:
            print('Ktargets = ', K, 'reached!')
            break
        
        # find index of max SSES for next bisection and remove from list of clusters
        print(SSES)
        ind = SSES.index(max(SSES))
        print('Remove cluster ',ind)
        cluster_selected.pop()
        cluster_selected = CLUSTERIND.pop(ind)
        SSES.pop(ind)
    
    return CLUSTERIND, INERTIAS

def DR_PCA(mat):
    pca = PCA(n_components=20)
    return csr_matrix(pca.fit_transform(mat.toarray()))

def DR_TSNE(mat):
    tsne = TSNE(n_components=2, perplexity=200, early_exaggeration=20, method='exact')
    return csr_matrix(tsne.fit_transform(mat))