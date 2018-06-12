# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:31:45 2018

@author: moseli
"""
from numpy.random import seed
seed(1)

from sklearn.model_selection import train_test_split as tts
import logging

from pyrouge import Rouge155 
import matplotlib.pyplot as plt
import keras
from keras import backend as k
k.set_learning_phase(1)
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Dense,LSTM,Input,Activation,Add,TimeDistributed,\
Permute,Flatten,RepeatVector,merge,Lambda,Multiply
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

#######################model params###########################
batch_size = 50
num_classes = 1
epochs = 20
hidden_units = emb_size_all
learning_rate = 0.005
clip_norm = 2.0

en_shape=np.shape(train_data["article"][0])
de_shape=np.shape(train_data["summaries"][0])

#######################################################################
############################Helper Functions###########################
#######################################################################    

def encoder_decoder(data):
    print('Encoder_Decoder LSTM...')
   
    """__encoder___"""
    encoder_inputs = Input(shape=en_shape)
    
    
    encoder_LSTM = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,return_state=True)
    encoder_LSTM_rev=LSTM(hidden_units,return_state=True,return_sequences=True,dropout_U=0.05,dropout_W=0.05,go_backwards=True)
    
  
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_inputs)
    encoder_outputsR, state_hR, state_cR = encoder_LSTM_rev(encoder_inputs)
    
    state_hfinal=Add()([state_h,state_hR])
    state_cfinal=Add()([state_c,state_cR])
    
    encoder_outputs_final = Add()([encoder_outputs,encoder_outputsR])
    attention = TimeDistributed(Dense(1, activation = 'tanh'))(encoder_outputs_final)

    encoder_states = [state_hfinal,state_cfinal]
    
    """____decoder___"""
    decoder_inputs = Input(shape=(None,de_shape[1]))
    decoder_LSTM = LSTM(hidden_units,return_sequences=True,dropout_U=0.2,dropout_W=0.2,return_state=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_inputs,initial_state=encoder_states)
    
    #Pull out XGBoost, (I mean attention)
    attention = Flatten()(attention)
    attention = Multiply()([decoder_outputs, attention])
    attention = Activation('softmax')(attention)
    attention = RepeatVector(hidden_units)(attention)
    attention = Permute([2, 1])(attention)
    sent_representation = Multiply()([decoder_outputs, attention])
    decoder_outputs = Lambda(lambda xin: k.sum(xin, axis=1),output_shape=(hidden_units,))(sent_representation)
    
    
    decoder_dense = Dense(de_shape[1],activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)
    print(model.summary())
    
    rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['accuracy'])
    
    x_train,x_test,y_train,y_test=tts(data["article"],data["summaries"],test_size=0.20)
    history= model.fit(x=[x_train,y_train],
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_test,y_test], y_test))
    print(model.summary())
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
    
    scores = model.evaluate([x_test,y_test],y_test, verbose=1)
    
    
    print('LSTM test scores:', scores)
    print('\007')
    print(model.summary())
    return model,encoder_model_inf,decoder_model_inf,history


"""_________generate summary from vectors_____________"""


def generateText(SentOfVecs):
    SentOfVecs=np.reshape(SentOfVecs,de_shape)
    kk=""
    for k in SentOfVecs:
        kk = kk + label_encoder.inverse_transform([argmax(k)])[0].strip()+" "
        #kk=kk+((getWord(k)[0]+" ") if getWord(k)[1]>0.01 else "")
    return kk

"""________________generate summary vectors___________"""

def summarize(article):
    stop_pred = False
    article =  np.reshape(article,(1,en_shape[0],en_shape[1]))
    #get initial h and c values from encoder
    init_state_val = encoder.predict(article)
    target_seq = np.zeros((1,1,de_shape[1]))
    #target_seq =np.reshape(train_data['summaries'][k][0],(1,1,de_shape[1]))
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

"""__________________Plot training curves_______________"""

def plot_training(history):
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def saveModels():
    trained_model.save("%sinit_model"%modelLocation)
    encoder.save("%sencoder"%modelLocation)
    decoder.save("%sdecoder"%modelLocation)
    
def evaluate_summ(article):
    ref=''
    for k in wt(data['summaries'][article])[:20]:
        ref=ref+' '+k
    gen_sum = generateText(summarize(train_data["article"][article]))
    print("-----------------------------------------------------")
    print("Original summary")
    print(ref)
    print("-----------------------------------------------------")
    print("Generated summary")
    print(gen_sum)
    print("-----------------------------------------------------")
    rouge = Rouge155()
    score = rouge.score_summary(ref, gen_sum)
    print("Rouge1 Score: ",score)
        
#######################################################################################
################################ Train model and test##################################
#######################################################################################

trained_model,encoder,decoder,history = encoder_decoder(train_data)
plot_training(history)
evaluate_summ(10)

print(generateText(summarize(train_data["article"][8])))
print(data["summaries"][8])
print(data["articles"][78])

