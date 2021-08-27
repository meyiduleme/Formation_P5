# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 15:14:44 2021

@author: dulem
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
nlp = spacy.load("en_core_web_sm")

def return_lang(labels):
    languages = [lang for lang in labels if lang in ['python','java','sql','r','javascript']] 
    return languages

def lst_to_str(lst):
    unpacked = ''.join(lst)
    return unpacked

def lemmatize(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    return tokens

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

def tfidf_features(X):
    """
        X — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    
    tfidf_vectorizer =  TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')#  '(\S+)'  means any no white space
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    return X_tfidf, tfidf_vectorizer.vocabulary_

def text_to_int(data, vocab):
   
    data_int = []
    for index in range(len(data)):
            data_temp = []
            for c in data[index]:
                if c in vocab:
                    data_temp.append(vocab[c])
            data_int.append(data_temp)
    return data_int