### Numpy implementation of topn/coreset/kmeans++

import numpy as np
import torch
import torch.nn.functional as F
import random


def topn(S, N):
    '''
    Implementation of topn selection
    :param S: (n,) uncertainty scores
    :param N: how many AL to be selected
    :return (N,) selected indices
    '''    
    # directly select topN images
    c_sets = np.asarray(range(len(S)))[np.argsort(S)[::-1][:N]]
    return c_sets


def kmeans_plus(S, feat, N):
    '''
    Implementation of kmeans++
    :param S: (n,) uncertainty scores
    :param feat: (n,f) features array
    :param N: how many AL to be selected
    :return (N,) selected indices
    '''
    
    S = np.asarray(S)
    feat = np.asarray(feat)
    c_sets = []
    
    # compute similarity matrix --> convert to cosine distance
    feat = F.normalize(torch.from_numpy(feat), dim=1, p=2).numpy()
    D = 1 - feat @ feat.T # nxn, this step would be slow if n is large
    D = np.clip(D, a_min=0., a_max=1.) # assure correct range
    print("Similarity computation finished")

    # randomly find first centroid
    c0 = random.sample(range(len(S)), 1)[0]
    c_sets.append(c0)
    
    # for loop until N centroids are found
    while len(c_sets) < N:
        # create mask so centroids will not be covered
        mask = np.ones((len(S),), bool)
        mask[np.asarray(c_sets)] = False # mask of shape (n,)

        # get distance w.r.t. nearest centroid
        D_select = D[mask, :][:, np.asarray(c_sets)]
        mind = np.min(D_select, axis=1) # mind of shape (n-c,)

        # sample according to probability distribution
        p_sample = mind*S[mask]/np.sum(mind*S[mask]) # p_sample of shape (n-c,)
        ci = np.random.choice(np.asarray(range(len(S)))[mask], 1, p=p_sample)[0]
        
        # add to centroid list
        c_sets.append(ci)
        if len(c_sets) % 100 == 0:
            print(len(c_sets))
        
    # return n centroids
    return c_sets

def core_set(S, feat, N):
    '''
    Implementation of core-set
    :param S: (n,) uncertainty scores
    :param feat: (n,f) features array
    :param N: how many AL to be selected
    :return (N,) selected indices
    '''
    
    S = np.asarray(S)
    feat = np.asarray(feat)
    c_sets = []
    
    # compute similarity matrix --> convert to cosine distance
    feat = F.normalize(torch.from_numpy(feat), dim=1, p=2).numpy()
    D = 1 - feat @ feat.T # nxn, this step would be slow if n is large
    D = np.clip(D, a_min=0., a_max=1.) # assure correct range
    print("Similarity computation finished")

    # randomly find first centroid
    c0 = random.sample(range(len(S)), 1)[0]
    c_sets.append(c0)
    
    # for loop until N centroids are found
    while len(c_sets) < N:
        # create mask so centroids will not be covered
        mask = np.ones((len(S),), bool)
        mask[np.asarray(c_sets)] = False # mask of shape (n,)

        # get distance w.r.t. nearest centroid
        D_select = D[mask, :][:, np.asarray(c_sets)]
        mind = np.min(D_select, axis=1) # mind of shape (n-c,)

        # select the hightest sample
        p_sample = mind*S[mask] # w/o normalization
        ci = np.asarray(range(len(S)))[mask][np.argsort(p_sample)[::-1][0]]
        
        # add to centroid list
        c_sets.append(ci)
        if len(c_sets) % 100 == 0:
            print(len(c_sets))
        
    # return n centroids
    return c_sets