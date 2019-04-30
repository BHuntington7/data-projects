# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 20:06:43 2019

@author: bhunt
"""

import bcolz
import numpy as np
import pickle

words = []

idx = 0

word2idx = {}

vectors = bcolz.carray(np.zeros(1), rootdir='storagedat', mode='w')


counter = 0
with open('vectors_pruned200.txt', 'rb') as f:

    for l in f:
        
        line = l.decode().split()

        word = line[0]

        words.append(word)

        word2idx[word] = idx

        idx += 1

        vect = np.array(line[1:]).astype(np.float)

        vectors.append(vect)
        
        if(counter<1):
            print(vect)
        counter = counter +1

    

vectors = bcolz.carray(vectors[1:].reshape((counter, 200)), rootdir='storagedat', mode='w')

vectors.flush()

pickle.dump(words, open('worddump.pkl', 'wb'))

pickle.dump(word2idx, open('vectordumpwordsID.pkl', 'wb'))