# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 23:34:50 2021

@author: dulem
"""

import joblib
import pickle

import numpy as np
from functions import *
from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st

#load pipeline tfidf vectorizer + logistic regression model

model = joblib.load('lr.joblib')
vectorizer = joblib.load('vectorizer.joblib') 


open_file = open("top_50_tags.pkl", "rb")
top_50_tags = np.array(pickle.load(open_file))
open_file.close()

def preprocessor(text):
    text_1, filtered_sentence = text_prepare(text)
    question = vectorizer.transform(filtered_sentence)
    return question

def classify_message(model, question):  
    prediction = model.predict_proba(question)
    tag = top_50_tags[prediction.mean(axis=0).argmax()]
    prob = prediction.mean(axis=0)[prediction.mean(axis=0).argmax()] 
    return {'label': tag, 'probability': prob}



#input text
st.write("# Automatic tagging Engine")
text = st.text_area("Enter your StackOverflow question")
#display labels 

if text != '':
   question = preprocessor(text)
   result = classify_message(model, question)
   st.write(result)