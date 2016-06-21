import sys
sys.path.append('../nn')
import Layers as Lay, Connections as Con, Activation
import cPickle as p
import InitParam as init
import numpy as np
##############################
# hyperparam

numEmbed = 100
numLSTM = 100
numOut = 6

# read vocabulary
fvoc = open('dic100.pkl', 'rb')
vocab = p.load(fvoc)
print 'vocabulary loaded'


##############################
# initialize weights
def generateParam(isEmbed = True):
    global Weights
    global Biases
    
    global B_i
    global B_f
    global B_o
    global B_g
    
    global W_i
    global W_f
    global W_o
    global W_g
    
    global U_i
    global U_f
    global U_o
    global U_g
    
    global Wout
    global Bout
    
    Weights = np.array([])
    Biases  = np.array([])

    if isEmbed == True:       
        Biases, Bembed = init.InitParam(Biases, num = numEmbed * len(vocab) )
    
    Biases, B_i = init.InitParam(Biases, num = numLSTM )
    Biases, B_f = init.InitParam(Biases, num = numLSTM )
    Biases, B_o = init.InitParam(Biases, num = numLSTM )
    Biases, B_g = init.InitParam(Biases, num = numLSTM )

    Weights, W_i = init.InitParam(Weights, num = numLSTM * numEmbed )
    Weights, W_f = init.InitParam(Weights, num = numLSTM * numEmbed )
    Weights, W_o = init.InitParam(Weights, num = numLSTM * numEmbed )
    Weights, W_g = init.InitParam(Weights, num = numLSTM * numEmbed )

    Weights, U_i = init.InitParam(Weights, num = numLSTM * numLSTM )
    Weights, U_f = init.InitParam(Weights, num = numLSTM * numLSTM )
    Weights, U_o = init.InitParam(Weights, num = numLSTM * numLSTM )
    Weights, U_g = init.InitParam(Weights, num = numLSTM * numLSTM )

    Weights, Wout = init.InitParam(Weights, num = numLSTM * numOut)
    Biases,  Bout = init.InitParam(Biases,  num = numOut)

    print '#weights =', len(Weights), '#biases =', len(Biases)
    
    return Weights, Biases

def lstm(sen):
    layers = []
    
    ###########################################
    # construct a layer for each word
    # Layers
    # |-----vector-----------| (nWords)
    # constructing a layer
    # def __init__(self, name, Bidx, numunit):
    layers = []
    
    for idx, w in enumerate(sen):
        if w in vocab:
            word_id = vocab[w] # d is the word
        else:
            word_id = 0
        embedLayer = Lay.layer( w, word_id * numEmbed, numEmbed, '0')
        
        i = Lay.layer( 'i_' + w, B_i[0], numLSTM, 'l' )
        f = Lay.layer( 'f_' + w, B_f[0], numLSTM, 'l' )
        o = Lay.layer( 'o_' + w, B_o[0], numLSTM, 'l' )
        g = Lay.layer( 'g_' + w, B_g[0], numLSTM, 'l' )
        
        
        
        # c_tilde is the c after applying activation function
        c = Lay.layer( 'c_' + w, -1,  numLSTM, '0' )
        c_tilde = Lay.layer( 'c_tilde_' + w, -1, numLSTM, 't' )

        h = Lay.layer( 'h_' + w, -1,  numLSTM, '0' )

        layers.append(embedLayer)
        
        layers.append( i )
        layers.append( f )
        layers.append( o )
        layers.append( g )
        
        layers.append( c )
        layers.append( c_tilde )
        
        layers.append( h )
    
        
        ########################
        # connections within this time slot
        # connection:
        #   def __init__(self, xlayer, ylayer, Widx, Wcoef = 1.0)
        # BilinearConnection:
        #   def __init__(self, xlayer1, xlayer2, ylayer, Widx)
        Con.connection(embedLayer, i, W_i[0])
        Con.connection(embedLayer, f, W_f[0])
        Con.connection(embedLayer, o, W_o[0])
        Con.connection(embedLayer, g, W_g[0])
        
        Con.BilinearConnection( i, g, c, -1 )
        Con.connection( c, c_tilde, -1)
        Con.BilinearConnection( o, c_tilde, h, -1)
        
        ########################
        # recurrent connections
        # layers[-9]: hidden layer of last time slot (h)
        # layers[-11]: cell of last time slot (c)
        if idx != 0:
            Con.connection(layers[-9], i, U_i[0])
            Con.connection(layers[-9], f, U_f[0])
            Con.connection(layers[-9], o, U_o[0])
            Con.connection(layers[-9], g, U_g[0])
            
            #print 'layer[-9].name: ', layers[-9].name
            #print 'layer[-11].name: ', layers[-11].name
            
            # self loop in c:
            Con.BilinearConnection( layers[-11], f, c, -1)
    
    ###########################
    # output layer
    # softmax
    
    outlayer = Lay.layer('output', Bout[0], numOut, 's')
    Con.connection(layers[-1], outlayer, Wout[0])
    
    layers.append(outlayer)
    
    return layers
'''
a list of word, current word id, head id
'''
if __name__ == '__main__':
    
    sentence = ['kids', 'play', 'in', 'yards']
                     
    W, B = generateParam(False)
    layers = lstm(sentence)
    #########################
    #  just check a little bit
    
    print 'Totally', len(layers), 'layer(s)'
    for l in layers:
        print l.name
        if l.lay_type != 'p':
            print '    bidx =', l.bidx
        print '    numunit =', l.numUnit
        
        print "    Up:"
        for c in l.connectUp:
            if (c.con_type == 'm' or c.con_type == '-' or c.con_type == 'b'):
                print "        ", c.xlayer1.name, " -> ", c.ylayer.name
                print "        ", c.xlayer2.name, " -> ", c.ylayer.name
            else:
                print "        ", c.xlayer.name, " -> ", c.ylayer.name
            
            if c.con_type != 'p':
                print '    widx =', c.Widx
      
        print "    Down:"
        for c in l.connectDown:
            if (c.con_type == 'm' or c.con_type == '-' or c.con_type == 'b'):
                print "        ", c.xlayer1.name, " -> ", c.ylayer.name
                print "        ", c.xlayer2.name, " -> ", c.ylayer.name
            else:
                print "        ", c.xlayer.name, " -> ", c.ylayer.name
            
            if c.con_type != 'p':
                print '    widx =', c.Widx
                
