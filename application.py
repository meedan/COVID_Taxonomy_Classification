#!/usr/bin/env python
# coding: utf-8

#import pickle, json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from stratified_kfold import create_train_test_splits, getstratifiedkfold
#import pandas as pd 
#import numpy as np
import os
#from transformers import DataCollatorWithPadding
#from transformers import TrainingArguments, Trainer
#from sklearn.metrics import classification_report
import copy

from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import requests
import json
from joblib import Memory
location = './cachedir'
disk_cache = Memory(location, verbose=0)

class Query(BaseModel):
    text: str


app = FastAPI()

origins = [
	#"http://localhost:3000",
	"https://coinsights.org",
	"https://demo.coinsights.org",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#from flask import Flask, request

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
	{"id":0, "label":"Distortions of Science", "title":"How were safe and effective vaccines developed so rapidly?", "url":"https://www.factcheck.org/scicheck_digest/did-the-speed-of-vaccine-development-compromise-on-safety/"},
	{"id":1, "label":"The Origins of COVID-19", "title":"What do we know about the origins of SARS-CoV-2?", "url":"https://www.factcheck.org/scicheck_digest/what-do-we-know-about-the-origins-of-sars-cov-2/"},
	{"id":2, "label":"Transmission","title":"How is COVID-19 transmitted?","url":"https://www.factcheck.org/scicheck_digest/how-is-covid-19-transmitted/"},
	{"id":3, "label":"The Nature, Existence, and Virulence of SARS-CoV-2","title":"How lethal is COVID-19?", "url":"https://www.factcheck.org/scicheck_digest/how-lethal-is-covid-19/"},
	{"id":4, "label":"Diagnosis and Tracing","title":"What tests are available for COVID-19?", "url":"https://www.factcheck.org/scicheck_digest/what-tests-are-available-for-covid-19/"},
	{"id":5, "label":"Prevention","title":"What evidence supports the use of face masks against the coronavirus?", "url":"https://www.factcheck.org/scicheck_digest/what-evidence-supports-the-use-of-face-masks-against-the-coronavirus/"},
	{"id":6, "label":"Preventatives and Treatment","title":"What treatments are available for COVID-19?", "url":"https://www.factcheck.org/scicheck_digest/what-treatments-are-available-for-covid-19/"},
	{"id":7, "label":"Vaccination","title":"The latest information about COVID-19 vaccinations","url":"https://www.factcheck.org/covid-misconceptions/#:~:text=is%20Industrial%20Bleach-,Vaccination,-SciCheck%20Digests"},
	{"id":8, "label":"Misrepresentation of Government Guidance","title":"What is the latest government guidance?","url":"https://www.factcheck.org/covid-misconceptions/#:~:text=Misrepresentation%20of%20Government%20Guidance"},
]
	

MODEL_DIR = "./models/CTBERT/fold_5e6_0/checkpoint-845"
topk = 3
TOKENIZER = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
model = AutoModelForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', num_labels=10)
model = load_model_from_checkpoint(MODEL_DIR, model)
#data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)


feed_headers={
"Accept": "application/vnd.api+json",
"X-Check-Token": open(".token").read().strip(),
}

@app.get("/default/coinsights-demo")
async def default_http(type: str = "feeds", categorizationQuery:str ="", feedsType:str="text", feedsQuery:str=""):
	if type=="feeds" and feedsQuery!="":
		feed_params={
			"filter[type]":feedsType,
			"filter[query]":feedsQuery
		}
		return feed_query_cache(feed_params)
	elif type=="categorization" and categorizationQuery!="":
		return {"data":infer_covid_category(categorizationQuery,topk)}
	else:
		return {"error":"Missing parameters."}

def feed_query_cache(feed_params):
	#if feed_params["filter[query]"] in FEED_CACHE:
	#	return FEED_CACHE[feed_params]
	#else:
	resp=requests.get("https://check-api.checkmedia.org/api/v2/feeds",params=feed_params,headers=feed_headers)
	resp={"data":resp.json()["data"]}
	#feed_params["filter[query]"]=resp
	return resp
feed_query_cache=disk_cache.cache(feed_query_cache)

#@app.route('/covid/categorize', methods = ["GET", "POST"])
@app.post("/covid/categorize")
async def infer_covid_category_http(query: Query):
	#if request.method == "GET":
	#	return json.dumps(OUTPUT_MAP)
	#elif request.method=="POST":
	#	data = request.get_json(force=True)
	#	text = data["text"]
	#	return infer_covid_category(text)
	return infer_covid_category(query.text,topk)

def infer_covid_category(text,num_results):
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
		return raw_output[0:num_results]

if __name__=="__main__":
	print("---------------------")
	print(infer_covid_category("the vaccine increases your chances of getting covid"))
	
