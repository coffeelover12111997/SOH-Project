# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:18:10 2017

@author: preetish
"""

#data extractor

import spacy
from nltk.tokenize import sent_tokenize
import numpy as np
import re
import pandas as pd

text=open('C:/Users/preetish/Downloads/soh/sentiment labelled sentences/amazon_cells_labelled.txt','r')
text1=open('C:/Users/preetish/Downloads/soh/sentiment labelled sentences/imdb_labelled.txt','r')
text2=open('C:/Users/preetish/Downloads/soh/sentiment labelled sentences/yelp_labelled.txt','r')

txt=''

for i in text:
    txt+=i

for j in text1:
    txt+=j

for k in text2:
    txt+=k

sent=sent_tokenize(txt)

senti=[]
c=0

for i in sent:
    if i[0]=='0' or i[0]=='1':
        senti.append(i[0])

temp=[]

temp.append(sent[0])

for i in sent:
    if i[0]=='0' or i[0]=='1':         
        temp.append(i)

temp=temp[:2958]

nlp=spacy.load('en')

for i in range(len(temp)):
    temp1=re.findall('[a-zA-Z ]',temp[i])
    temp1=''.join(temp1)
    temp1=[i.text for i in nlp(temp1) if i.pos_ != 'NOUN']
    temp[i]=' '.join(temp1)
    
temp=np.array(temp,dtype='str')  
senti=np.array(senti,dtype='int')

final=np.append(temp.reshape(-1,1),senti.reshape(-1,1),axis=1)
final=pd.DataFrame(final)

final.to_csv('data2.csv')

