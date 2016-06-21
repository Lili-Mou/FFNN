import sys
import types
sys.path.append('../nn')

import construct_senlstm as Construct
import cPickle as p


import serialize

'''
def refine(dependencies):
    cnt = 0
    mapping = dict()
    for i, [_, idx,_,_] in enumerate( dependencies):
        #_, idx, _, _ = dependencies[i]
        if idx in mapping:
            dependencies[i][1] = mapping[idx]
            continue
        mapping[idx] = cnt
        dependencies[i][1] = cnt
        cnt += 1
        
    for i, [_, _, idx, _] in enumerate(dependencies):
        if idx != 0:
            dependencies[i][2] = mapping[idx]
        else:
            dependencies[i][2] = -1
        
    pass
'''
def clean_word( word):
    while len(word)>0 and not word[0].isalpha():
        word = word[1:]
    while len(word)>0 and not word[-1].isalpha():
        word = word[:-1]
    return word
def clean_word_in_dep( dependencies):
    for d in dependencies:
        d[0] = clean_word( d[0] )
cnt = 1
if __name__ == '__main__':
    toprocess = ['train', 'test','cv']
    #toprocess = ['test']
    Construct.generateParam(True)
    fout = None
    for this_file in toprocess:
        print "processing: ", this_file
        cnt = 1
        if fout != None:
            fout.close()
        fout = None
        
        sampleid = 0
        fin1 = open('raw_data/'+ this_file + '.txt')
        
        while True:
            sentence_A = fin1.readline().strip().lower()
            
            if (not sentence_A):
                print cnt - 1
                fin1.close()
                break
            
            sen_A = sentence_A.split()
            
            layers = Construct.lstm(sen_A)
            serialize.serialize(layers, 'nets/net_'+this_file+'/'+str(cnt - 1))


            if cnt % 100 == 0:
                print cnt
            cnt += 1
