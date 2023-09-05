# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 12:30:07 2023
Inference for Character Sequence to Sequence translation
URL : https://keras.io/examples/nlp/lstm_seq2seq/
@author: kulje

Problems and Solutions :
        P1 : ValueError: Found input tensor cannot be reached given provided output tensors.
        S1 : Cannot reuse a variable if that variable is a Keras Input
            (encode_inputs was assigned twice )
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

latent_dim = 256 # for FRA - Latent dimensionality of the encoding space
num_samples = 10000 # Number of samples to train
data_path = "C:\\Kuljeet\\WorkSpace\\NLP\\jpn-eng\\jpn.txt"

#Vectorize the data
input_texts = []
target_texts = []
num_encoder_tokens = 0 
num_decoder_tokens = 0
input_token_index = {}
target_token_index = {}
max_decoder_seq_len = 0 

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
    
    global input_token_index
    global target_token_index
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

def load_model(saved_model):
    model = keras.models.load_model(saved_model)
    print(model.summary())
    encoder_inputs = model.inputs[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output #lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs,encoder_states)
    
    decoder_inputs =  model.inputs[1]
    decoder_state_input_h = keras.Input(shape=(latent_dim,))
    decoder_state_input_c = keras.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm( decoder_inputs, 
                                                             initial_state = decoder_states_inputs)
    decoder_states = [state_h_dec,state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, 
                                [decoder_outputs]+decoder_states)
    
    #reverse_lookup token index to decodes sequences back to readable characters
    reverse_input_char_index = dict((i,char) for char,i in input_token_index.items())
    reverse_target_char_index = dict((i,char) for char,i in target_token_index.items())
    return reverse_input_char_index, reverse_target_char_index, encoder_model, decoder_model 

def decode_sequence(input_seq, encoder_model, decoder_model, reverse_target_char_index ):
    #encode input as states vector
    states_value = encoder_model.predict(input_seq)
    #generate empty target sequnce of length 1
    target_seq= np.zeros((1,1,num_decoder_tokens))
    #populate first character of the target sequence with the start charcter
    target_seq[0,0, target_token_index["\t"]] = 1.0
    # sampling loop for batch of sequences (batch_size assumed 1) 
    stop_condition =False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq]+states_value)
    
        #sample a token
        sampled_token_index =  np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
    
        #Exit criteria = max_length or stop char
        if sampled_char=="\n" or len(decoded_sentence) > max_decoder_seq_len :
            stop_condition = True
        
        #Update the target sequence of length 1
        target_seq = np.zeros((1,1,num_decoder_tokens))
        target_seq[0,0, sampled_token_index] = 1.0
        
        #updates states
        states_value = [h, c]
    return decoded_sentence


def main():
    encoder_input_data,decoder_input_data,decoder_target_data = load_data()
    reverse_input_char_index, reverse_target_char_index, encoder_model, decoder_model = load_model("cs2s")
    for seq_index in range(20):
        # translate sequences one at a time
        input_seq = encoder_input_data[seq_index : seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, reverse_target_char_index)
        print("================")
        print("Input Seq :",input_texts[seq_index])
        print("Decoded Seq :", decoded_sentence)
    return        
       
#main function
if __name__ == '__main__' :
    main()        