#!/usr/bin/env python
# coding: utf-8

import pickle, json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from stratified_kfold import create_train_test_splits, getstratifiedkfold
import pandas as pd 
import numpy as np
import os
#from datasets import Dataset, Value, ClassLabel, Features
from transformers import DataCollatorWithPadding


def tokenize_function(entry):
	#a = TOKENIZER(entry['text'])
	a = TOKENIZER(entry)
	return {'input_ids':a['input_ids']} #, 'labels': entry['cat']}


#from datasets import load_metric
from transformers import TrainingArguments, Trainer
from sklearn.metrics import classification_report

def compute_metrics(eval_preds):
	metric1 = load_metric("accuracy")
	metric2 = load_metric("f1")
	logits, labels = eval_preds
	predictions = np.argmax(logits, axis=-1)
	return {'Accuracy': metric1.compute(predictions=predictions, references=labels)['accuracy'],
			'F1': metric2.compute(predictions=predictions, references=labels, average=None)['f1'].tolist(),
			'Weighted F1': metric2.compute(predictions=predictions, references=labels, average='weighted')['f1']}

def load_model_from_checkpoint(path, model):
	state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'),map_location=torch.device('cpu'))
	model.load_state_dict(state_dict)
	return model

def get_best_checkpoint(input_dir):
	checkpoints = os.listdir(input_dir)
	checkpoints = [f for f in checkpoints if 'checkpoint' in f ]
	max_checkpoint_num = max([int(f.split('-')[-1]) for f in checkpoints])
	state_file = input_dir+'checkpoint-'+str(max_checkpoint_num)+'/trainer_state.json'
	with open(state_file, 'r') as f:
		state = json.loads(f.read())
	best_checkpoint = state['best_model_checkpoint']
	return best_checkpoint

#https://www.factcheck.org/covid-misconceptions/
OUTPUT_MAP=[
	{"index":0, "label":"Scientific Findings/Misleading Attacks on Public Health Scientists","url":"https://www.factcheck.org/scicheck_digest/did-the-speed-of-vaccine-development-compromise-on-safety/"},
	{"index":1, "label":"The Origins of COVID-19","url":"https://www.factcheck.org/scicheck_digest/what-do-we-know-about-the-origins-of-sars-cov-2/"},
	{"index":2, "label":"Transmission","url":"https://www.factcheck.org/scicheck_digest/how-is-covid-19-transmitted/"},
	{"index":3, "label":"The Nature, Existence, and Virulence of SARS-CoV-2","url":"https://www.factcheck.org/scicheck_digest/how-lethal-is-covid-19/"},
	{"index":4, "label":"Diagnosis and Tracing","url":"https://www.factcheck.org/scicheck_digest/what-tests-are-available-for-covid-19/"},
	{"index":5, "label":"Prevention (3 W’s, ventilation, lockdowns/quarantine)","url":"https://www.factcheck.org/scicheck_digest/what-evidence-supports-the-use-of-face-masks-against-the-coronavirus/"},
	{"index":6, "label":"Preventatives and Treatment","url":"https://www.factcheck.org/scicheck_digest/what-treatments-are-available-for-covid-19/"},
	{"index":7, "label":"Vaccination","url":"#vaccination#"}, #TODO: This category needs one summarizing article!
	{"index":8, "label":"Misrepresentation of Constitutional Protections or Government Guidance","url":"https://www.factcheck.org/scicheck_digest/could-a-covid-19-vaccine-become-mandatory/"},
]
	

base_model_dir = './models/CTBERT/fold_5e6_'
topk = 3
TOKENIZER = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
model = AutoModelForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', num_labels=10)


#>>> get_best_checkpoint("./models/CTBERT/fold_5e6_0/")
#'./models/CTBERT/fold_5e6_0/checkpoint-845'
#>>> get_best_checkpoint("./models/CTBERT/fold_5e6_1/")
#'./models/CTBERT/fold_5e6_1/checkpoint-1352'
#>>> get_best_checkpoint("./models/CTBERT/fold_5e6_2/")
#'./models/CTBERT/fold_5e6_2/checkpoint-1014'
#>>> get_best_checkpoint("./models/CTBERT/fold_5e6_3/")
#'./models/CTBERT/fold_5e6_3/checkpoint-1690'
#>>> get_best_checkpoint("./models/CTBERT/fold_5e6_4/")
#'./models/CTBERT/fold_5e6_4/checkpoint-1837'

fold_num=0 #TODO: We should train will all folds
seed_dir = base_model_dir+str(fold_num)+'/'
best_checkpoint = get_best_checkpoint(seed_dir)
checkpoint_id = best_checkpoint.split('/')[-1]
best_checkpoint = seed_dir+checkpoint_id

model = load_model_from_checkpoint(best_checkpoint, model)
data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)


query_input=["the vaccine increases your chances of getting covid", "eat alkaline foods to prevent covid", "More vaccinated than unvaccinated people are dying from COVID","the virus isn't real", "The virus is caused by 5G"]
tokenized_input=map(tokenize_function,query_input)

with torch.no_grad():
	preds, trues, probs = [], [], []
	for i, data in enumerate(tokenized_input):
		print("---------------------")
		print("query:", query_input[i])
		inputs = torch.tensor([data['input_ids']])
		outputs = model(inputs)
		print("Top match:", list(map(lambda x: OUTPUT_MAP[x]["label"],outputs.logits.argmax(-1).tolist())))
		#print(outputs.logits.tolist())
		preds = preds + outputs.logits.argmax(-1).tolist()
		probs = probs + outputs.logits.tolist()
		
		preds_k = []
		for p in probs:
			#top_k=sorted(range(len(p)), key=lambda i: p[i])[-topk:]
			top_k=sorted(range(len(p)), key=lambda i: -p[i])[0:topk]
			preds_k.append(top_k)
		print("Top k:", list(map(lambda x: OUTPUT_MAP[x]["label"],top_k)))

print("---------------------")		
#print(classification_report(labels, preds, target_names=[i for i in 'ABCDEFGHIK']))
	
#category_wise_recall = recall_at_k(all_trues, all_preds_k)
#print(category_wise_recall)
