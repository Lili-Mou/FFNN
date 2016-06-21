#    embeddings = p.load(open('word_vector_cache_large.pkl','rb') )
  
#    Biases, Bembed = init.InitParam(Biases, newWeights = embeddings)


import numpy as np
import sys
import cPickle as p
sys.path.append('../nn/')
sys.path.append('../')
import FFNN
import InitParam as init
import copy
import serialize, write_param
import struct, process

construct =  process.Construct

# read data
import read_data as read

X_train, X_test, X_CV, y_train, y_test, y_CV = read.readXy()


#np.random.seed(200)
#np.random.shuffle(X_train)#
#np.random.seed(200)
#np.random.shuffle(y_train)

#np.random.seed(100)
#np.random.shuffle(X_test)
#np.random.seed(100)
#np.random.shuffle(y_test)

'''X_CV = X_train[-400:]
y_CV = y_train[-400:]
X_train = X_train[:-400]
y_train = y_train[:-400]
'''

#y_train.extend(y_train)
#X_train_copy = [ x + '_aug' for x in X_train]
#X_train.extend(X_train_copy)

np.random.seed(200)
np.random.shuffle(X_train)
np.random.seed(200)
np.random.shuffle(y_train)


def write_nets_in_one_file(X, path, foutname):
    print 'writing ', foutname
    fout = file(foutname, 'wb')
    
    for num, x in enumerate(X):
        if num % 100 == 0:
            print num
       # if num == 10000:
       #    break
        tmp = struct.pack('i', num )
        #print num	
        fout.write( tmp )
        fin = file(path + str(X[num]), 'rb')
        tmpstr = fin.read()
        if len(tmpstr) <= 10:
            print "great error! num = ", num, "; file = ", X[num]
        fout.write( tmpstr )
        fin.close()
        
    fout.close()

print "writing samples"
fingerprint = 'QC_LSTM'

write_nets_in_one_file(X_train, './', 'preprocessed/train_nets_'+fingerprint)
write_nets_in_one_file(X_CV,    './', 'preprocessed/CV_nets_'+ fingerprint )
write_nets_in_one_file(X_test,  './', 'preprocessed/test_nets_' + fingerprint)

serialize.write_labels('preprocessed/labels', y_train, y_CV, y_test)
print "writing para"



##############################
# initialize weights

#np.random.seed(123)
np.random.seed(435)
#np.random.seed(126)
#np.random.seed(127)
#np.random.seed(787)
    
Weights, Biases = construct.generateParam(False)
    
Weights = Weights.reshape((-1,1))
Biases  = Biases.reshape((-1,1))   
print len(Weights), len(Biases)

write_param.write_binary('preprocessed/para_'+fingerprint, Weights, Biases)

print 'done!'

