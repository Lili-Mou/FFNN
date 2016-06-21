import os
import numpy as np


def readXy():
    X_train = []
    X_test  = []
    X_CV    = []
    y_train = []
    y_test  = []
    y_CV    = []
    ####################################
    # ytrain    
    tmpy = []
    fy = open('raw_data/y_train.txt')
    #fy = open('original_data/y_train.txt')
    while True:
        line = fy.readline()
        if not line:
            break
        line = line.strip()
        tmpy.append( float(line) )
    
    
    tmpy = np.array(tmpy)
    print 'tmpy', len(tmpy)
 
    y_train_orig = np.asarray(tmpy, dtype=np.int)
    
    
    
    ####################################
    # y test
    tmpy = []
    fy = open('raw_data/y_test.txt')
    #fy = open('original_data/y_test.txt')
    while True:
        line = fy.readline()
        if not line:
            break
        line = line.strip()
        tmpy.append( float(line) )
    
    
    tmpy = np.array(tmpy)
    print 'tmpy', len(tmpy)
    y_test_orig = np.asarray(tmpy, dtype=np.int)
    
    ####################################
    # y CV
    tmpy = []
    fy = open('raw_data/y_cv.txt')
    #fy = open('original_data/y_test.txt')
    while True:
        line = fy.readline()
        if not line:
            break
        line = line.strip()
        tmpy.append( float(line) )
    
    
    tmpy = np.array(tmpy)
    print 'tmpy', len(tmpy)
    y_CV_orig = np.asarray(tmpy, dtype=np.int)
    #####################################
    # X_train
    
    filedir = 'nets/net_train/'
    length = len( y_train_orig )
    for i in xrange(length):
        onefile = filedir + str(i) 
        if os.path.exists( onefile ):
            X_train.append( onefile )
            y_train.append( y_train_orig[i] )
                        

    ###################################
    filedir = 'nets/net_test/'
    length = len( y_test_orig )
    for i in xrange(length):
        onefile = filedir + str(i) 
       
	if os.path.exists( onefile ):
            X_test.append( onefile )
            y_test.append( y_test_orig[i] )
    ###################################
    filedir = 'nets/net_cv/'
    length = len( y_CV_orig )
    for i in xrange(length):
        onefile = filedir + str(i) 
       
	if os.path.exists( onefile ):
            X_CV.append( onefile )
            y_CV.append( y_CV_orig[i] )
 

    return X_train, X_test,X_CV, y_train, y_test,y_CV
    

if __name__ == '__main__':
    X_small, y_small = readSmall()
    print len(X_small), len(y_small)
    print y_small[0], y_small[-1]
    X_train, X_CV, X_test, y_train, y_CV, y_test = readXy()
