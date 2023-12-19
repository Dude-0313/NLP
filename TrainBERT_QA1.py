# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:08:46 2023
Training a BERT QA Model
https://keras.io/examples/nlp/question_answering/
@author: kulje

Problems & Solutions :
    P1 : RuntimeError: 
            An attempt has been made to start a new process before the
            current process has finished its bootstrapping phase.
    S1 : Any python code using multiprocessing with  fork/spawn requires the 
        implementation wrapped in a function call
        Wrapped the call to datasets.map() in main() and if __name__ == '__main__'
    P2 : AttributeError: module '__main__' has no attribute '__spec__'
    S2 : Code to be execute from outside Spider without below segment 
            __spec__ = None
            with Pool() as mp_pool:
                main()
    P3 : While loading object
            ValueError: Unknown loss function: dummy_loss. Please ensure this 
            object is passed to the `custom_objects` argument
    S3 : Load the model with compile=False and then comple the model separately
    P4 : AttributeError: 'dict' object has no attribute 'start_logits'
    S4 : No solution found yet, unable to load the saved model
"""
import tensorflow as tf
from datasets import load_dataset
from tensorflow import keras
from transformers import AutoTokenizer
from transformers import TFAutoModelForQuestionAnswering

datasets= load_dataset("squad")

print(datasets["train"][0])
print(datasets["validation"][0])

model_checkpoint = "distilbert-base-cased"
# Auto classes(as AutoTokenizer here) loads appropriate versions of classes based on the name
# and checkpoint values passed (e.g. bert-base-uncased will load BertTokeizer and gpt2 will load
# GPT2Tokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 384  # The maximum length of a feature (question and context)
doc_stride =(128)  # The authorized overlap between two part of the context when splitting

# Tokenize our examples with truncation and padding, but keep the overflows using a
# stride. This results in one example possible giving several features when a context is long,
# each of those features having a context that overlaps a bit the context of the previous
# feature.
def prepare_train_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]    
    examples["context"] = [c.lstrip() for c in examples["context"]]    
    tokenized_examples = tokenizer( examples["question"], examples["context"],
                                   truncation="only_second",
                                   max_length=max_length,
                                   stride=doc_stride,
                                   return_overflowing_tokens=True, # To get the list of features capped by the maximum length
                                   return_offsets_mapping=True, #To see which feature of the original context contain the answer
                                   padding="max_length")
    # Since one example might give us several features if it has a long context, we need a
    # map from a feature to its corresponding example. This key gives us just that.
    sample_mapping=tokenized_examples.pop("overflow_to_sample_mapping")    
    # The offset mappings will give us a map from token to character position in the original
    # context. This will help us compute the start_positions and end_positions.
    offset_mapping=tokenized_examples.pop("offset_mapping")  
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"]=[]
    
    for i, offsets in enumerate(offset_mapping):
       # In the case of impossible answers (the answer is in another feature given by an example with a long context), 
       # we set the cls index for both the start and end position. 
        input_ids = tokenized_examples["input_ids"][i]
        cls_index =input_ids.index(tokenizer.cls_token_id)
        # Grab the sequence corresponding to that example (to know what is the context and what
        # is the question).    
        sequence_ids =tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this
        # span of text.        
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"])==0 :
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else :
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # Start token index of the current span in the text.
            token_start_index=0
            while sequence_ids[token_start_index] !=1:
                token_start_index+=1
            
            token_end_index = len(input_ids) - 1    
            while sequence_ids[token_end_index]!=1:
                token_end_index -=1
                
            #Detect if feature is out of the span {label with CLS if it is}
            if not (
                    offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char ):
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index 
                while(
                        token_start_index <len(offsets) and
                        offsets[token_start_index][0] <= start_char) :
                    token_start_index += 1
                tokenized_examples['start_positions'].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char :
                    token_end_index -=1
                tokenized_examples['end_positions'].append(token_end_index + 1)
    return tokenized_examples


def main(train):
    if train == True :
        # apply on all sentences of our dataset using map
        tokenized_datasets = datasets.map(
            prepare_train_features,
            batched=True,
            remove_columns=datasets['train'].column_names,
            num_proc=3)
        #load complete dataset as a numpy arrays
        train_set = tokenized_datasets['train'].with_format('numpy')[:]
        validation_set =  tokenized_datasets['validation'].with_format('numpy')[:]
        # from_Pretrained customizes the pre trained model by removing the language model 
        # and adds a new head for question answering task
        model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
        
        #can achive better performance wing the learning rate decay and decoupled weight decay
        optimizer =  keras.optimizers.Adam(learning_rate=5e-5)
        
        # all transformer models come with their own (default) loss and can be changed if so needed
        keras.mixed_precision.set_global_policy("mixed_float16")
        
        model.compile(optimizer=optimizer)
        
        #train model , note that seperate labels are not passed but are keyed in the training dataset
        model.fit(train_set,validation_data=validation_set)

    # Save and Load is not working ...
        #P4 model.save("bertQAmodel")
    #P4 else :
        #Test
        #P4 model=keras.models.load_model("bertQAmodel",compile=False)
        #can achive better performance wing the learning rate decay and decoupled weight decay
        #P4 optimizer =  keras.optimizers.Adam(learning_rate=5e-5)
        #P4  model.compile(optimizer=optimizer)

        context = """Keras is an API designed for human beings, not machines. Keras follows best
        practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes
        the number of user actions required for common use cases, and it provides clear &
        actionable error messages. It also has extensive documentation and developer guides. """
        question = "What is Keras?"
        
    # Mystery : Both of below lines produce exacly same results , but Why ?     
    #    inputs = tokenizer([context],[question],return_tensors="np")
        inputs = tokenizer([question],[context],return_tensors="np")
        outputs = model(inputs)
        start_position =  tf.argmax(outputs.start_logits,axis=1)
        end_position= tf.argmax(outputs.end_logits,axis=1)
        
        print(int(start_position),int(end_position[0]))
        
        answer = inputs["input_ids"][0,int(start_position) : int(end_position)+1]
        print(answer)
        
        print(tokenizer.decode(answer))
    return

from multiprocessing import Pool
if __name__ == '__main__':
    __spec__ = None
    with Pool() as mp_pool:
        main(True)
    