# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:06:04 2017

@author: preetish
"""

#soh
'''
'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015,
twitter data from sanders analytics'''


from gensim.models.keyedvectors import KeyedVectors
import spacy
import pandas as pd
import re
import numpy as np
from sklearn.svm import SVC
import pickle

ftext=KeyedVectors.load_word2vec_format('C:/Users/preetish/Downloads/soh/crawl-300d-2M.vec',binary=False)
#data=pd.read_csv('C:/Users/preetish/Downloads/soh/train.txt',sep='\n',encoding='latin-1',header=None)




data1=pd.read_csv('C:/Users/preetish/AppData/Roaming/SPB_16.6/data1.csv')
data2=pd.read_csv('C:/Users/preetish/AppData/Roaming/SPB_16.6/data2.csv')
data3=pd.read_csv('C:/Users/preetish/AppData/Roaming/SPB_16.6/data3.csv')
data4=pd.read_csv('C:/Users/preetish/AppData/Roaming/SPB_16.6/data4.csv')



data=pd.read_csv('brightfuture.csv',sep=';',error_bad_lines=False)



def word2vec(sentarray):
    vecrep=[]
    nlp=spacy.load('en')
    
    for i in sentarray:
        v=np.zeros((300,))
        for j in nlp(i):
            try:
                v+=ftext.word_vec(j.text)
            except KeyError:
                v+=np.zeros((300,))
        vecrep.append(v/len(i))
    return vecrep
    

tr2=np.array(word2vec(data2['0'].astype(str)))
tr3=np.array(word2vec(data3['sentence']))
tr4=np.array(word2vec(data4['sentence']))

test=np.array(word2vec(data['text']))


tr2=np.append(tr2,np.array(data2['1']).reshape(-1,1),axis=1)
tr3=np.append(tr3,np.array(data3['senti']).reshape(-1,1),axis=1)
tr4=np.append(tr4,np.array(data4['senti']).reshape(-1,1),axis=1)


t2=np.append(tr2,tr3,axis=0)
#t2=np.append(t2,tr4,axis=0)
np.random.shuffle(t2)


trainX=t2[:,:300]
testX=tr4[:,:300]
trainy=t2[:,300]
testy=tr4[:,300]

model=SVC(C=1,gamma=3)
model.fit(trainX,trainy)
print(model.score(trainX,trainy))
print(model.score(testX,testy))

var=[]
final=[]

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))


    


