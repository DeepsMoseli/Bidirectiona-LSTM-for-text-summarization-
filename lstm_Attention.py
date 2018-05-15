# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:31:45 2018

@author: moseli
"""
from numpy.random import seed
seed(1)

from sklearn.model_selection import train_test_split as tts
import logging

import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd


import keras
from keras import backend as k
k.set_learning_phase(1)
from keras.preprocessing.text import Tokenizer
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,Concatenate
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
#keras.utils.vis_utils import plot_model


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

#######################model params###########################
batch_size = 10
num_classes = 1
epochs = 10
hidden_units = emb_size_all
learning_rate = 0.008
clip_norm = 2.0

###############################################################
en_shape=np.shape(train_data["article"][0])
de_shape=np.shape(train_data["summaries"][0])

"""______generate summary length for test______"""
#train_data["nums_summ"]=list(map(lambda x:0 if len(x)<5000 else 1,data["articles"]))
#train_data["nums_summ"]=list(map(len,data["summaries"]))
#train_data["nums_summ_norm"]=(np.array(train_data["nums_summ"])-min(train_data["nums_summ"]))/(max(train_data["nums_summ"])-min(train_data["nums_summ"]))

############################helpers###########################

def encoder_decoder(data):
    print('Encoder_Decoder LSTM...')
   
    """__encoder___"""
    encoder_inputs = Input(shape=en_shape)
    
    encoder_LSTM = LSTM(hidden_units, dropout_U = 0.2, dropout_W = 0.2 ,return_state=True)
    encoder_LSTM_rev=LSTM(hidden_units,return_state=True,go_backwards=True)
    
    #merger=Add()[encoder_LSTM(encoder_inputs), encoder_LSTM_rev(encoder_inputs)]
    encoder_outputsR, state_hR, state_cR = encoder_LSTM_rev(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_inputs)
    
    state_hfinal=Add()([state_h,state_hR])
    state_cfinal=Add()([state_c,state_cR])
    
    encoder_states = [state_hfinal,state_cfinal]
    
    """____decoder___"""
    decoder_inputs = Input(shape=(None,de_shape[1]))
    decoder_LSTM = LSTM(hidden_units,return_sequences=True,return_state=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_inputs,initial_state=encoder_states) 
    decoder_dense = Dense(de_shape[1],activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)
    #plot_model(model, to_file=modelLocation+'model.png', show_shapes=True)
    rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    
    model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['accuracy'])

    x_train,x_test,y_train,y_test=tts(data["article"],data["summaries"],test_size=0.20)
    model.fit(x=[x_train,y_train],
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_test,y_test], y_test))
    
    """_________________inference mode__________________"""
    encoder_model_inf = Model(encoder_inputs,encoder_states)
    
    decoder_state_input_H = Input(shape=(hidden_units,))
    decoder_state_input_C = Input(shape=(hidden_units,)) 
    decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(decoder_inputs,
                                                                     initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model_inf= Model([decoder_inputs]+decoder_state_inputs,
                         [decoder_outputs]+decoder_states)
    
    
    #plot_model(encoder_model_inf, to_file='encoder_model.png', show_shapes=True)
    #plot_model(decoder_model_inf, to_file='decoder_model.png', show_shapes=True)
    scores = model.evaluate([x_test,y_test],y_test, verbose=0)
    
    
    print('LSTM test scores:', scores)
    #announcedone()
    print('\007')
    print(model.summary())
    return model,encoder_model_inf,decoder_model_inf

"""___pred____"""
def comparePred(index):
    pred=trained_model.predict([np.reshape(train_data["article"][index],(1,en_shape[0],emb_size_all)),np.reshape(train_data["summaries"][index],(1,de_shape[0],emb_size_all))])
    return pred


"""____generate summary from vectors and remove padding words___"""
def generateText(SentOfVecs):
    SentOfVecs=np.reshape(SentOfVecs,de_shape)
    kk=""
    for k in SentOfVecs:
        kk = kk + label_encoder.inverse_transform([argmax(k)])[0].strip()+" "
        #kk=kk+((getWord(k)[0]+" ") if getWord(k)[1]>0.01 else "")
    return kk

"""___generate summary vectors___"""

def summarize(article):
    stop_pred = False
    article =  np.reshape(article,(1,en_shape[0],en_shape[1]))
    
    #get initial h and c values from encoder
    init_state_val = encoder.predict(article)
    target_seq = np.zeros((1,1,de_shape[1]))
    
    generated_summary=[]
    while not stop_pred:
        decoder_out,decoder_h,decoder_c= decoder.predict(x=[target_seq]+init_state_val)
        generated_summary.append(decoder_out)
        init_state_val= [decoder_h,decoder_c]
        #get most similar word and put in line to be input in next timestep
        #target_seq=np.reshape(model.wv[getWord(decoder_out)[0]],(1,1,emb_size_all))
        target_seq=np.reshape(decoder_out,(1,1,de_shape[1]))
        if len(generated_summary)== de_shape[0]:
            stop_pred=True
            break
    return generated_summary
        
        
#######################################################################################

trained_model,encoder,decoder = encoder_decoder(train_data)

def saveModels():
    trained_model.save("%sinit_model"%modelLocation)
    encoder.save("%sencoder"%modelLocation)
    decoder.save("%sdecoder"%modelLocation)

print(generateText(summarize(train_data["article"][1])))
print(data["summaries"][1])
print(data["articles"][1])

del trained_model,encoder,decoder


getWord(collect_pred[23])
model.wv.most_similar(np.zeros((1,emb_size_all)))
