# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:16:05 2018

@author: moseli
"""

"""

Sequence of code:
    1) cnn_daily_load.py
    2) word2vec.py
    3) lstm.py
    
"""
import winsound as ws
import numpy as np
import os
import pandas as pd
import re


#########################################################
#####################define data sources#################
#########################################################
CNN_data="C:\\Users\\moseli\\Documents\\Masters of Information technology\\Masters project\\text mining\\data\\new CNN\\cnn\\"
daily_data="C:\\Users\\moseli\\Documents\\Masters of Information technology\\Masters project\\text mining\\data\\new CNN\\dailymail\\"

datasets={"cnn":CNN_data,"dailymail":daily_data}

data_categories=["training","validation","test"]

data={"articles":[],"summaries":[]}

#########################################################
##################helpers################################
########################################################


def parsetext(dire,category,filename):
    with open("%s\\%s"%(dire+category,filename),'r',encoding="Latin-1") as readin:
        print("file read successfully")
        text=readin.read()
    return text.lower()


def load_data(dire,category):
    """dataname refers to either training, test or validation"""
    for dirs,subdr, files in os.walk(dire+category):
        filenames=files
    return filenames


def cleantext(text):
    text=re.sub(r"what's","what is ",text)
    text=re.sub(r"it's","it is ",text)
    text=re.sub(r"\'ve"," have ",text)
    text=re.sub(r"i'm","i am ",text)
    text=re.sub(r"\'re"," are ",text)
    text=re.sub(r"n't"," not ",text)
    text=re.sub(r"\'d"," would ",text)
    text=re.sub(r"\'s","s",text)
    text=re.sub(r"\'ll"," will ",text)
    text=re.sub(r"can't"," cannot ",text)
    text=re.sub(r" e g "," eg ",text)
    text=re.sub(r"e-mail","email",text)
    text=re.sub(r"9\\/11"," 911 ",text)
    text=re.sub(r" u.s"," american ",text)
    text=re.sub(r" u.n"," united nations ",text)
    text=re.sub(r"\n"," ",text)
    text=re.sub(r":"," ",text)
    text=re.sub(r"-"," ",text)
    text=re.sub(r"\_"," ",text)
    text=re.sub(r"\d+"," ",text)
    text=re.sub(r"[$#@%&*!~?%{}()]"," ",text)
    
    return text

def printArticlesum(k):
    print("---------------------original sentence-----------------------")
    print("-------------------------------------------------------------")
    print(data["articles"][k])
    print("----------------------Summary sentence-----------------------")
    print("-------------------------------------------------------------")
    print(data["summaries"][k])
    return 0


def announcedone():
    duration=2000
    freq=440
    ws.Beep(freq,duration)


###########################################################

filenames=load_data(datasets["cnn"],data_categories[0])

"""----------load the data, sentences and summaries-----------"""
for k in range(len(filenames[:400])):
        if k%2==0:
            try:
                data["articles"].append(cleantext(parsetext(datasets["cnn"],data_categories[0],"%s"%filenames[k])))
            except Exception as e:
                data["articles"].append("Could not read")
                print(e)
        else:
            try:
                data["summaries"].append(cleantext(parsetext(datasets["cnn"],data_categories[0],"%s"%filenames[k])))
            except Exception as e:
                data["summaries"].append("Could not read")
                print(e)

del filenames
#printArticlesum(30)
