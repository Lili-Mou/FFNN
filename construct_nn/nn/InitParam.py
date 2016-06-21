# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:46:46 2014

@author: mou
"""

import numpy as np

def InitMatrixParam(OldWeights, num=None, n_in=None, n_out=None):

    oldlen = len(OldWeights)    
    
    for i in xrange(num):
        #lower = -np.sqrt(6. / (n_in + n_out))
        #upper =  np.sqrt(6. / (n_in + n_out))
        lower = -.8
        upper = .8
        tmpWeights = np.random.uniform(low=lower, high=upper, size=(n_in, n_out))
        OldWeights = np.concatenate((OldWeights, tmpWeights.reshape(-1)))
        
    return OldWeights, range(oldlen, oldlen+num*n_in*n_out)


def InitParam(OldWeights, num = None, newWeights = None, upper = None, lower = None):

    oldlen = len(OldWeights)    
    if newWeights != None:
        newWeights = np.array(newWeights)    
        num = len(newWeights)
        OldWeights = np.concatenate((OldWeights, newWeights.reshape(-1) ))
        
    else:
        if upper == None:
            upper = -.02
            lower = 0.02
        tmpWeights = np.random.normal(0, 0.02, num)
        OldWeights = np.concatenate((OldWeights, tmpWeights.reshape(-1) ))
    return OldWeights,range(oldlen, oldlen + num)


if __name__ == '__main__':
    Weights = []
    Weights, idx1 = InitParam(Weights, 10 )
    Weights, idx2 =  InitParam(Weights, 5)
    print Weights, len(Weights)
    Weights, idx3 = InitParam(Weights, newWeights=[1,2,3])
    print idx1
    print idx2

