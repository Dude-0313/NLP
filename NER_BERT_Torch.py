# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:49:19 2023
Named Entity Recognition using PyTorch
URL : https://wandb.ai/mostafaibrahim17/ml-articles/reports/Named-Entity-Recognition-With-HuggingFace-Using-PyTorch-and-W-B--Vmlldzo0NDgzODA2
Dataset : https://www.kaggle.com/datasets/juliangarratt/conll2003-dataset
@author: kulje
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

#globals
train_file = "C:\\Kuljeet\\WorkSpace\\NLP\\CoNLLL2003_dataset\\conll2003\\eng.train"
validation_file = "C:\\Kuljeet\\WorkSpace\\NLP\\CoNLLL2003_dataset\\conll2003\\eng.testa"
test_file = "C:\\Kuljeet\\WorkSpace\\NLP\\CoNLLL2003_dataset\\conll2003\\eng.testb"

model_name = "bert-base-cased"

def checkCUDA():
    print("CUDA available",torch.cuda.is_available())
    print("Current available index :", torch.cuda.current_device())
    print("Device name : ",torch.cuda.get_device_name(torch.cuda.current_device()))
    return

def load_data(filepath):
    with open(filepath,"r") as f:
        content = f.read().strip()
        sentences = content.split("\n\n")
        data = []
        for sentence in sentences :
            tokens = sentence.split("\n")
            token_data = []
            for token in tokens:
                token_data.append(token.split())
            data.append(token_data)
    return data

#convert to Hugging Face Dataset format
def convert_to_dataset(data, label_map):
    formatted_data = {"tokens":[],"ner_tags":[]}
    for sentence in data:
        tokens = [token_data[0] for token_data in sentence]
        ner_tags = [label_map[token_data[3]] for token_data in sentence]
        formatted_data["tokens"].append(tokens)
        formatted_data["ner_tags"].append(ner_tags)
    return Dataset.from_dict(formatted_data)


    

def main():
    checkCUDA()
    train_data = load_data(train_file)
    validation_data = load_data(validation_file)
    test_data = load_data(test_file)
    label_list = sorted(list(set([token_data[3] for sentence in train_data for token_data in sentence])))
    label_map = {label : i for i, label in enumerate(label_list)}

    train_dataset = convert_to_dataset(train_data,label_map)    
    validation_dataset = convert_to_dataset(validation_data,label_map)    
    test_dataset = convert_to_dataset(test_data,label_map)    
    
    datasets = DatasetDict({
        "train": train_dataset,
        "validation" : validation_dataset,
        "test" : test_dataset,
        })
    
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    model =  AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

    #define metrics function 
    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        predictions = np.argmax(predictions,axis=2)
        
        #remove ignored indexes (special_tokens)
        true_predictions = [
            [label_list[p] for (p,l) in zip(prediction, label) if l != -100]
            for prediction,label in zip(predictions,labels)
            ]        
        
        true_labels = [
            [label_list[l] for (p,l) in zip(prediction, label) if l != -100]
            for prediction,label in zip(predictions,labels)
            ]

        return {
            "precision" : precision_score(true_labels, true_predictions),
            "recall" : recall_score(true_labels,true_predictions),
            "f1" : f1_score(true_labels,true_predictions),
            "classification_report" : classification_report(true_labels, true_predictions)
            }    

    #Tokenize inputs and align with labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"],truncation=True,
                                     is_split_into_words=True, padding=True)
        labels=[]
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None :
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx 
            labels.append(label_ids)
        tokenized_inputs["labels"]=labels
        return tokenized_inputs

    #Takes batches of data and coverts to tensors adding padding as required
    def data_collator(data):
        input_ids = [torch.tensor(item["input_ids"]) for item in data]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in data]
        labels = [torch.tensor(item["labels"]) for item in data]
    
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask,batch_first=True,padding_value=0)            
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,padding_value=-100)
        
        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "labels":labels
            }
        
    tokenized_datasets = datasets.map(tokenize_and_align_labels,batched=True)

        #define training parametrs using HuggungFace's Tarining APIs for PyTorch                
    training_args = TrainingArguments(
        output_dir=".\\results",
        evaluation_strategy = "steps",
        eval_steps=500,
        save_steps=500,
        num_train_epochs=1,
        per_device_eval_batch_size=8,
        per_gpu_eval_batch_size=8,
        logging_steps=100,
        learning_rate=1e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1")

    trainer = Trainer(
       model=model,
       args = training_args,
       train_dataset= tokenized_datasets["train"],
       eval_dataset= tokenized_datasets["validation"],
       data_collator= data_collator,
       tokenizer=tokenizer,
       compute_metrics= compute_metrics
       )
    
    # start training
    trainer.train()
    
    #test
    sentence = "Wipro Limited is one of the largest IT services company in India and provides customized Software and Hardware solutions to fortune 500 companies."
    tokenized_inputs = tokenizer(sentence,return_tensors="pt").to(model.device)
    outputs = model(**tokenized_inputs)
    predicted_labels = outputs.logits.argmax(-1)[0]
    named_entities = [tokenizer.decode([token]) for token, label in 
                      zip(tokenized_inputs["input_ids"][0], predicted_labels) 
                      if label !=0 and label != label_map['O']]
                
    print("NERs :",named_entities)
    return
    
if __name__ == "__main__" :
    main()