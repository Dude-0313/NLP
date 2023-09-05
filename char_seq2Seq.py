# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:05:00 2023
Character Sequence to Sequence translation
URL : https://keras.io/examples/nlp/lstm_seq2seq/
@author: kulje

Problems & Solutions:
    P1 : globle variable assignment not reflected after function call
    S1 : use 'global redefinition' within the function

"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

batch_size = 64
epochs = 100
latent_dim = 256 # for FRA - Latent dimensionality of the encoding space
num_samples = 10000 # Number of samples to train
#data_path = "C:\\Kuljeet\\WorkSpace\\NLP\\fra-eng\\fra.txt"
data_path = "C:\\Kuljeet\\WorkSpace\\NLP\\jpn-eng\\jpn.txt"

#Vectorize the data
input_texts = []
target_texts = []
num_encoder_tokens = 0 
num_decoder_tokens = 0

def load_data():
    input_characters = set()
    target_characters = set()
    with open(data_path,"r",encoding='utf-8') as f :
        lines = f.read().split("\n")
    for line in lines[: min(num_samples,len(lines)-1)]:
        input_text,target_text, _ = line.split("\t")
        # use tab as start_seq and "\n" as end_seq
        target_text= "\t" + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text :
            if char not in input_characters :
              input_characters.add(char) 
        for char in target_text :
            if char not in target_characters :
              target_characters.add(char) 
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    
    global num_encoder_tokens 
    global num_decoder_tokens 
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_len = max([len(txt) for txt in input_texts])
    max_decoder_seq_len = max([len(txt) for txt in target_texts])
    
    input_token_index =  dict([(char,i) for i, char in enumerate(input_characters)])
    target_token_index =  dict([(char,i) for i, char in enumerate(target_characters)])
    
    encoder_input_data = np.zeros((len(input_texts),max_encoder_seq_len,num_encoder_tokens),dtype="float32")
    decoder_input_data = np.zeros((len(input_texts),max_decoder_seq_len,num_decoder_tokens),dtype="float32")
    decoder_target_data = np.zeros((len(input_texts),max_decoder_seq_len,num_decoder_tokens),dtype="float32")
    
    for i, (input_text,target_text) in enumerate(zip(input_texts, target_texts)) :
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]]=1.0
        encoder_input_data[ i, t+1:, input_token_index[" "]] =1.0
        for t, char in enumerate(target_text):
            #decoder_target_data is ahead of the decoder_input_data by one timestamp
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t >0 :
                #decoder_target_data is ahead by one timestamp
                # and will not include the start character
                decoder_target_data[i,t-1,target_token_index[char]]=1.0
        decoder_input_data[i, t+1:, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0
  
    return encoder_input_data,decoder_input_data,decoder_target_data

def build_model(encoder_input_data,decoder_input_data,decoder_target_data):
    encoder_inputs = keras.Input(shape=(None,num_encoder_tokens))
    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    #Discard 'encoder_outputs' and keep the states
    encoder_states = [state_h,state_c]
    
    #Setup the decoder, using 'encoder_states' as initial state
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

    # return states to be used during inference later
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True) 
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model =keras.Model([encoder_inputs, decoder_inputs],decoder_outputs)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",metrics=['accuracy'])
    print(model.summary())
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./TBLogs")
    model.fit([encoder_input_data,decoder_input_data], decoder_target_data, 
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[tensorboard_callback])
    #Save model
    model.save("cs2s")
    
def main():
    encoder_input_data,decoder_input_data,decoder_target_data = load_data()
    build_model(encoder_input_data,decoder_input_data,decoder_target_data)


#main function
if __name__ == '__main__' :
    main()