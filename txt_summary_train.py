# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:09:38 2023
Abstractive Text Summarization 
URL : https://keras.io/examples/nlp/t5_hf_summarization/
Dataset : https://huggingface.co/datasets/xsum

@author: kulje
"""

import os
import logging
#import nltk
#import numpy as np
import tensorflow as tf
import keras_nlp
from tensorflow import keras
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers.keras_callbacks import KerasMetricCallback
from transformers import pipeline
#Set log level
tf.get_logger().setLevel(logging.ERROR)

#do not allow parallel tokenizer -- see issue in TestBERT_QA
os.environ["TOKENIZER_PARALLELISM"]="false"

# Set globals
TRAIN_TEST_SPLIT = 0.1
MAX_INPUT_LENGTH = 1024
MIN_TARGET_LENGTH = 5 # min output length
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_EPOCHS = 1

# Hugging-face pretrained model to use for Abstractive Summarization
# Text-To-Text Transfer Transformer (T5)
MODEL_CHECKPOINT = "t5-small"


def load_data():
    raw_datasets = load_dataset("xsum",split="train")
    print(raw_datasets)
    raw_datasets= raw_datasets.train_test_split(test_size=TRAIN_TEST_SPLIT)
    return raw_datasets


def pre_process(raw_datasets):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    # to tell the model to summarize, it can also translate
    if MODEL_CHECKPOINT in ["t5-small","t5-base","t5-large","t5-3b","t5-11b"]:
        prefix = "summarize: "
    else:
        prefix = ""
    # Tokenize the text dataset (input and targets) into it's corresponding token ids that will be used for embedding look-up in BERT
    # Add the prefix to the tokens
    # Create additional inputs for the model like token_type_ids, attention_mask, etc.
    def preprocess_function(examples):
       inputs = [ prefix + doc for doc in examples["document"]] 
       model_inputs = tokenizer(inputs,max_length= MAX_INPUT_LENGTH, truncation=True)
       # tokenize targets
       with tokenizer.as_target_tokenizer():
           labels = tokenizer(examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True)
           
       model_inputs["labels"]=labels["input_ids"]
       return model_inputs

    tokenized_datasets = raw_datasets.map(preprocess_function,batched=True)
    return tokenized_datasets, tokenizer

# using TFAutoModelForSeq2SeqLM as its a sequence to sequence task
def define_model():
    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer)
    return model

def prepare_datasets(tokenized_datasets,data_collator):
    train_dataset = tokenized_datasets["train"].to_tf_dataset(
        batch_size=BATCH_SIZE,
        columns=["input_ids","attention_mask","labels"],
        shuffle=True,
        collate_fn=data_collator)
    test_dataset = tokenized_datasets["test"].to_tf_dataset(
        batch_size=BATCH_SIZE,
        columns=["input_ids","attention_mask","labels"],
        shuffle=False,
        collate_fn=data_collator)
    # create generate_dataset to evaluale ROUGE Score on the fly
    generate_dataset = (tokenized_datasets["test"]
                        .shuffle()
                        .select(list(range(200)))
                        .to_tf_dataset(batch_size=BATCH_SIZE,
                         columns=["input_ids","attention_mask","labels"],
                         shuffle=False,
                         collate_fn=data_collator))
    
    return train_dataset,test_dataset,generate_dataset


def main():
    do_train = True
    raw_datasets = load_data()
    tokenized_datasets, tokenizer= pre_process(raw_datasets)
    model = define_model()
    data_collator = DataCollatorForSeq2Seq(tokenizer,model=model,return_tensors="tf")
    train_dataset,test_dataset,generate_dataset = prepare_datasets(tokenized_datasets, data_collator)
    rouge_l = keras_nlp.metrics.RougeL()
    def metric_fn(eval_predictions):
        predictions, labels = eval_predictions
        decoded_predictions = tokenizer.batch_decode( predictions,
                                                     skip_special_tokens=True)
        for label in labels :
            label[label<0] = tokenizer.pad_token_id # Replace masked label tokens
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True)
        result = rouge_l(decoded_labels,decoded_predictions)
        result = {"RougeL": result["f1_score"]}
        return result
    #define metric callback
    metric_callback = KerasMetricCallback(
        metric_fn, eval_dataset=generate_dataset,predict_with_generate=True)
    callbacks = [metric_callback]
    #train model
    if do_train == True :
        model.fit(train_dataset, validation_data = test_dataset,
                  epochs=MAX_EPOCHS, callbacks=callbacks)
    
    # Inference - use HuggingFace Summerization Pipeline
    # HuggingFace transformer pipelines makes it easier to use models for
    # inference, Task specific pipelines are available for use. Here we are
    # using the model we just trained for "summerization" task
    summerizer = pipeline("summerization", model=model
                          ,tokenizer=tokenizer, framework="tf")
    
    summerizer(raw_datasets["test"][0]["document"],
               min_length=MIN_TARGET_LENGTH,
               max_length=MAX_TARGET_LENGTH)

if __name__ == '__main__':
    main()    