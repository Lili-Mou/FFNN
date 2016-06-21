import numpy as np
import sys
import cPickle as p
sys.path.append('../')
import FFNN
import InitParam as init
import copy, Activation as act
import struct

def write_labels(fname, y_train, y_CV, y_test):
    f = file(fname, 'w')
    f.write( str(len(y_train)) + '\n')
   # f.write('10000\n')
    f.write( str(len(y_CV))    + '\n')
    f.write( str(len(y_test))  + '\n')
    '''    
    for i in X_train:
        f.write(str(i) + '\n')
    for i in X_CV:
        f.write(str(i) + '\n')
    for i in X_test:
        f.write(str(i) + '\n')
    '''
    cnt=0
    
    for i in y_train:
        cnt+=1
        f.write(str(i) + '\n')
   #     if cnt== 10000:
  #          break
    for i in y_CV:
        f.write(str(i) + '\n')
    for i in y_test:
        f.write(str(i) + '\n')

    f.close()


    
    
def serialize(layers, fname):
    
    f = file( fname, 'wb' )
    num_lay = struct.pack('i', len(layers) )
    if num_lay <=2 :
        print error
    f.write( num_lay)
    
    num_con = 0
    
    #################################
    # preprocessing, compute some indexes
    for i, layer in enumerate(layers):
        layer.idx = i
        num_con += len(layer.connectDown )
        for (icon, con) in enumerate(layer.connectDown):
            con.ydownid = icon
            
    num_con = struct.pack('i', num_con )
    f.write( num_con )
    
    #################################
    # layers
    #################################
    
    for layer in layers:
        
        lay_type = layer.lay_type

        # layer type
        tmp = struct.pack('c', lay_type)
        f.write( tmp )
        
        ##############################
        # pooling
        if lay_type == 'p':
            # numUnit        
            tmp = struct.pack('i', layer.numUnit )
            f.write( tmp )
            # numDown
            tmp = struct.pack('i', len(layer.connectDown) )
            f.write( tmp )
            # pool type
            if layer.poolType == 'max':
                tlayer = 'x'
            elif layer.poolType == 'sum':
                tlayer = 's'
            tmp = struct.pack('c', tlayer)
            f.write( tmp )
            
            # remark
            
            tmp = struct.pack('c', layer.remark)
            f.write( tmp )
        elif lay_type == 'a':
            # numUnit        
            tmp = struct.pack('i', layer.numUnit )
            f.write( tmp )
            # numDown
            tmp = struct.pack('i', len(layer.connectDown) )
            f.write( tmp )
            # activation
            tmp = struct.pack('c', layer.act)
            f.write( tmp )
            # bidx
            bidx = -1
            if layer.bidx != None:
                bidx = layer.bidx

            tmp = struct.pack('i', bidx)

            f.write(tmp)
            # remark
            #print layer.remark
            tmp = struct.pack('c', layer.remark)
            f.write( tmp )
    #########################
    # connections
    for layer in layers:
        for ydownid, con in enumerate(layer.connectDown):
            
            con_type = con.con_type

            tmp = struct.pack('c', con_type)
            f.write( tmp )

            if ydownid != con.ydownid:
                print 'great error', ydownid, con.ydownid
                asdf
            if con_type == 'l': # linear combinition
                # xlayer idx   
                tmp = struct.pack('i', con.xlayer.idx)
                f.write( tmp )
                # ylayer idx
                tmp = struct.pack('i', con.ylayer.idx )
                f.write( tmp)
                # idx in y's connectDown
                tmp = struct.pack('i', con.ydownid)
                f.write( tmp )
                # Widx
                tmp = struct.pack('i', con.Widx)
                f.write(tmp)
                # Wcoef
                tmp = struct.pack('f', con.Wcoef)
                f.write( tmp )
                    
            elif con_type == 'b':

                # xlayer_1 idx   
                tmp = struct.pack('i', con.xlayer1.idx)
                f.write( tmp )
                # xlayer_2 idx
                tmp = struct.pack('i', con.xlayer2.idx)
                f.write(tmp)
                # ylayer idx
                tmp = struct.pack('i', con.ylayer.idx )
                f.write( tmp)
                # idx in y's connectDown
                tmp = struct.pack('i', con.ydownid)
                f.write( tmp )
                # Widx
                Widx = con.Widx
                tmp = struct.pack('i', Widx)
                f.write( tmp )
                # Wcoef
                tmp = struct.pack('f', con.Wcoef)
                f.write(tmp)
                
            elif con_type == 'p': 
                # xlayer idx   
                tmp = struct.pack('i', con.xlayer.idx)
                f.write( tmp )
                # ylayer idx
                tmp = struct.pack('i', con.ylayer.idx )
                f.write( tmp)
                # idx in y's connectDown
                tmp = struct.pack('i', con.ydownid)
                f.write( tmp )    
            else:
            	print "con_type not recognized", con_type            
    f.close()
