# -*- coding: utf-8 -*-
#/usr/local/lib/python3.6


import os
import pickle
import nltk
import re
import csv
import spacy
import collections
from collections import Counter
import nltk


#Reading the directory of (.txt) files
directory = []
directory = os.listdir('/home/francielle/Fake.br/size_normalized_texts/fake')
os.chdir('/home/francielle/Fake.br/size_normalized_texts/fake')


nlp_pt = spacy.load('pt_core_news_sm')
for file in sorted(directory):
    open_file = open(file,'r', encoding='utf-8')
    text = open_file.read()
    doc = nlp_pt(text)
    
    #NER occurence
    cont_ner = []
    for ent in doc.ents:
        if (ent.label_ == 'PER'):
            cont_ner.append(ent.text)
    print(len(cont_ner))

    #Word occurence removing stop words
    words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    word_freq = Counter(words)
    common_words = word_freq.most_common(5)
    
    #Dependence tree parse analysis
    for token in doc:
        print(token.text, ';', token.dep_, ';', token.head.text, ';', token.head.pos_,';',
            [child for child in token.children])



            

