# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 23:34:50 2021

@author: dulem
"""

import joblib
import pickle

import numpy as np

import streamlit as st
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.tokenize import word_tokenize
#load pipeline tfidf vectorizer + logistic regression model

model = joblib.load('lr.joblib')
vectorizer = joblib.load('vectorizer.joblib') 


open_file = open("top_50_tags.pkl", "rb")
top_50_tags = np.array(pickle.load(open_file))
open_file.close()

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    STOPWORDS = set(stopwords.words('english'))
    # find all the urls
    pattern_url = re.compile(r'http.+?(?="|<)')
    #text = # lowercase text
    text =text.lower()
    # replace url by space in text
    text = re.sub(pattern_url, ' ', text)
    # lemmatize text
    #token_word = nlp(text)
    # delete stopwords from text
    token_word=word_tokenize(text)
    filtered_sentence = [w for w in token_word if not w in STOPWORDS] # filtered_sentence contain all words that are not in stopwords dictionary
    lenght_of_string=len(filtered_sentence)
    text_new=""
    for w in filtered_sentence:
        if w!=filtered_sentence[lenght_of_string-1]:
             text_new=text_new+w+" " # when w is not the last word separate by whitespace
        else:
            text_new=text_new+w
            
    text = text_new
    return text, filtered_sentence

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