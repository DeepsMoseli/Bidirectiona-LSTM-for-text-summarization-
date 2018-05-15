# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 21:59:33 2018

@author: moseli
"""
from numpy.random import seed
seed(1)

modelLocation="C:/Users/moseli/Documents/Masters of Information technology/Masters project/text mining/TrainedModels/"


import gensim as gs
import pandas as pd
import numpy as np
import scipy as sc
import nltk
from nltk.tokenize import word_tokenize as wt
from nltk.tokenize import sent_tokenize as st
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
import logging
import re
from collections import Counter



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

########################################################################
############################helpers####################################
#######################################################################
emb_size_all = 128
maxcorp=5000


def createCorpus(t):
    corpus = []
    all_sent = []
    for k in t:
        for p in t[k]:
            corpus.append(st(p))
    for sent in range(len(corpus)):
        for k in corpus[sent]:
            all_sent.append(k)
    for m in range(len(all_sent)):
        all_sent[m] = wt(all_sent[m])
    
    all_words=[]
    for sent in all_sent:
        hold=[]
        for word in sent:
            hold.append(word.lower())
        all_words.append(hold)
    return all_words


def word2vecmodel(corpus):
    emb_size = emb_size_all
    model_type={"skip_gram":1,"CBOW":0}
    window=10
    workers=4
    min_count=4
    batch_words=20
    epochs=25
    #include bigrams
    #bigramer = gs.models.Phrases(corpus)

    model=gs.models.Word2Vec(corpus,size=emb_size,sg=model_type["skip_gram"],
                             compute_loss=True,window=window,min_count=min_count,workers=workers,
                             batch_words=batch_words)
        
    model.train(corpus,total_examples=len(corpus),epochs=epochs)
    model.save("%sWord2vec"%modelLocation)
    print('\007')
    return model


def summonehot(corpus):
    allwords=[]
    annotated={}
    for sent in corpus:
        for word in wt(sent):
            allwords.append(word.lower())
    print(len(set(allwords)), "unique characters in corpus")
    #maxcorp=int(input("Enter desired number of vocabulary: "))
    maxcorp=int(len(set(allwords))/1.1)
    wordcount = Counter(allwords).most_common(maxcorp)
    allwords=[]
    
    for p in wordcount:
        allwords.append(p[0])  
        
    allwords=list(set(allwords))
    
    print(len(allwords), "unique characters in corpus after max corpus cut")
    #integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(allwords)
    #one hot
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #make look up dict
    for k in range(len(onehot_encoded)): 
        inverted = cleantext(label_encoder.inverse_transform([argmax(onehot_encoded[k, :])])[0]).strip()
        annotated[inverted]=onehot_encoded[k]
    return label_encoder,onehot_encoded,annotated



def wordvecmatrix(model,data):
    IO_data={"article":[],"summaries":[]}
    i=1
    for k in range(len(data["articles"])):
        art=[]
        summ=[]
        for word in wt(data["articles"][k].lower()):
            try:
                art.append(model.wv.word_vec(word))
            except Exception as e:
                print(e)

        for word in wt(data["summaries"][k].lower()):
            try:
                summ.append(onehot[word])
                #summ.append(model.wv.word_vec(word))
            except Exception as e:
                print(e)
        
        IO_data["article"].append(art) 
        IO_data["summaries"].append(summ)
        if i%100==0:
            print("progress: " + str(((i*100)/len(data["articles"]))))
        i+=1
    #announcedone()
    print('\007')
    return IO_data

def cutoffSequences(data,artLen,sumlen):
    data2={"article":[],"summaries":[]}
    for k in range(len(data["article"])):
        if len(data["article"][k])<artLen or len(data["summaries"][k])<sumlen:
             #data["article"]=np.delete(data["article"],k,0)
             #data["article"]=np.delete(data["summaries"],k,0)
             pass
        else:
            data2["article"].append(data["article"][k][:artLen])
            data2["summaries"].append(data["summaries"][k][:sumlen])
    return data2


def max_len(data):
    lenk=[]
    for k in data:
        lenk.append(len(k))
    print("The minimum length is: ",min(lenk))
    print("The average length is: ",np.average(lenk))
    print("The max length is: ",max(lenk))
    return min(lenk),max(lenk)

"""reshape vectres for Gensim"""
def reshape(vec):
    return np.reshape(vec,(1,emb_size_all))

def addones(seq):
    return np.insert(seq, [0], [[0],], axis = 0)

def endseq(seq):
    pp=len(seq)
    return np.insert(seq, [pp], [[1],], axis = 0)
#######################################################################
#######################################################################

corpus = createCorpus(data)

label_encoder,onehot_encoded,onehot=summonehot(data["summaries"])

model=word2vecmodel(corpus)

model.get_latest_training_loss()

train_data = wordvecmatrix(model,data)

print(len(train_data["article"]), "training articles")

train_data=cutoffSequences(train_data,300,25)

#seq length stats
max_len(train_data["article"])
max_len(train_data["summaries"])


train_data["summaries"]=np.array(train_data["summaries"])
train_data["article"]=np.array(train_data["article"])


#add end sequence for each article

#train_data["summaries"]=np.array(list(map(endseq,train_data["summaries"])))
#train_data["article"]=np.array(list(map(endseq,train_data["article"])))

print("summary length: ",len(train_data["summaries"][0]))
print("article length: ",len(train_data["article"][0]))


"""__pad sequences__
train_data["article"]=pad_sequences(train_data["article"],maxlen=max_len(train_data["article"]),
          padding='post',dtype=float)
train_data["summaries"]=pad_sequences(train_data["summaries"],maxlen=max_len(train_data["summaries"]),
          padding='post',dtype=float)
"""
#add start sequence
train_data["summaries"]=np.array(list(map(addones,train_data["summaries"])))

