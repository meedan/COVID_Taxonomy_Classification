#!/usr/bin/env python
# coding: utf-8

import pickle, json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from stratified_kfold import create_train_test_splits, getstratifiedkfold
import pandas as pd 
import numpy as np
import os
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from sklearn.metrics import classification_report
import copy

#from flask import Flask, url_for, render_template, Response, request

#app = Flask(__name__)

def tokenize_function(entry):
	a = TOKENIZER(entry)
	return {'input_ids':a['input_ids']} #, 'labels': entry['cat']}



def load_model_from_checkpoint(path, model):
	state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'),map_location=torch.device('cpu'))
	model.load_state_dict(state_dict)
	return model

#https://www.factcheck.org/covid-misconceptions/
OUTPUT_MAP=[
	{"id":0, "label":"Scientific Findings/Misleading Attacks on Public Health Scientists","url":"https://www.factcheck.org/scicheck_digest/did-the-speed-of-vaccine-development-compromise-on-safety/"},
	{"id":1, "label":"The Origins of COVID-19","url":"https://www.factcheck.org/scicheck_digest/what-do-we-know-about-the-origins-of-sars-cov-2/"},
	{"id":2, "label":"Transmission","url":"https://www.factcheck.org/scicheck_digest/how-is-covid-19-transmitted/"},
	{"id":3, "label":"The Nature, Existence, and Virulence of SARS-CoV-2","url":"https://www.factcheck.org/scicheck_digest/how-lethal-is-covid-19/"},
	{"id":4, "label":"Diagnosis and Tracing","url":"https://www.factcheck.org/scicheck_digest/what-tests-are-available-for-covid-19/"},
	{"id":5, "label":"Prevention (3 W’s, ventilation, lockdowns/quarantine)","url":"https://www.factcheck.org/scicheck_digest/what-evidence-supports-the-use-of-face-masks-against-the-coronavirus/"},
	{"id":6, "label":"Preventatives and Treatment","url":"https://www.factcheck.org/scicheck_digest/what-treatments-are-available-for-covid-19/"},
	{"id":7, "label":"Vaccination","url":"#vaccination#"}, #TODO: This category needs one summarizing article!
	{"id":8, "label":"Misrepresentation of Constitutional Protections or Government Guidance","url":"https://www.factcheck.org/scicheck_digest/could-a-covid-19-vaccine-become-mandatory/"},
]
	

MODEL_DIR = "./models/CTBERT/fold_5e6_0/checkpoint-845"
topk = 3
TOKENIZER = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
model = AutoModelForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', num_labels=10)
model = load_model_from_checkpoint(MODEL_DIR, model)
data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)

#@app.route('/covid/categorize')
def infer_covid_category(text):
	#query_input=["the vaccine increases your chances of getting covid", "eat alkaline foods to prevent covid", "More vaccinated than unvaccinated people are dying from COVID","the virus isn't real", "The virus is caused by 5G"]
	tokenized_input=tokenize_function(text)
	with torch.no_grad():
		print("query:", text)
		inputs = torch.tensor([tokenized_input['input_ids']])
		outputs = model(inputs)
		#print("Top match:", list(map(lambda x: OUTPUT_MAP[x]["label"],outputs.logits.argmax(-1).tolist())))
		#print(outputs.logits.tolist())
		#preds = outputs.logits.argmax(-1).tolist()
		#probs = outputs.logits.tolist()
		probs=outputs.logits.tolist()[0]
		raw_output=copy.deepcopy(OUTPUT_MAP)
		for entry,prob in zip(raw_output,probs):
			entry["probability"]=prob
		#Sort by probability
		raw_output.sort(reverse=True,key=lambda x:x["probability"])
		return json.dumps(raw_output)

print("---------------------")
print(infer_covid_category("the vaccine increases your chances of getting covid"))
