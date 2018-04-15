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
from keras.preprocessing.sequence import pad_sequences
import logging
import re


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

########################################################################
############################helpers####################################
#######################################################################

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
    emb_size=128
    model_type={"skip_gram":1,"CBOW":0}
    window=5
    workers=4
    min_count=3
    batch_words=75
    epochs=20
    #include bigrams
    bigramer = gs.models.Phrases(corpus)

    model=gs.models.Word2Vec(bigramer[corpus],size=emb_size,sg=model_type["skip_gram"],
                             compute_loss=True,window=window,min_count=min_count,workers=workers,
                             batch_words=batch_words)
        
    model.train(corpus,total_examples=len(corpus),epochs=epochs)
    model.save("%sWord2vec"%modelLocation)
    print('\007')
    return model
    
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
                summ.append(model.wv.word_vec(word))
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


def max_len(data):
    lenk=[]
    for k in data:
        lenk.append(len(k))
    return max(lenk)

"""reshape vectres for Gensim"""
def reshape(vec):
    return np.reshape(vec,(1,128))

"""__find nearest word to given vec__"""
def getWord(vec):
    word=model.wv.most_similar(reshape(vec),topn=1)[0]
    return word

def addones(seq):
    return np.insert(seq, [0], [[0],], axis = 0)

def endseq(seq):
    pp=len(seq)
    return np.insert(seq, [pp], [[1],], axis = 0)
#######################################################################
#######################################################################


corpus = createCorpus(data)

model=word2vecmodel(corpus)
model.get_latest_training_loss()

train_data = wordvecmatrix(model,data)

#add end sequence for each article
train_data["summaries"]=np.array(list(map(endseq,train_data["summaries"])))
train_data["article"]=np.array(list(map(endseq,train_data["article"])))

"""__pad sequences__"""
train_data["article"]=pad_sequences(train_data["article"],maxlen=max_len(train_data["article"]),
          padding='post',dtype=float)
train_data["summaries"]=pad_sequences(train_data["summaries"],maxlen=max_len(train_data["summaries"]),
          padding='post',dtype=float)

#add start sequence
train_data["summaries"]=np.array(list(map(addones,train_data["summaries"])))

