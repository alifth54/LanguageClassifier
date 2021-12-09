# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:15:46 2021

@author: ali

Discribe:
    In this project i want to train a model that can recognize the language of a text
    between 5 classes such English,presian,Frech,italian and spanish
"""
#%% imports
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print('using tensorflow version 2.7.0')
#%% import data
trainDf = pd.read_csv('comments.csv',
                      header=None)
#%% preprocessing and one hot coding  
trainDf.columns = ['sentence','lang']
Ncls = len(trainDf.lang.unique())

encoder = LabelEncoder()
target = trainDf['lang']
target = encoder.fit_transform(target)
target = tf.keras.utils.to_categorical(target,num_classes=Ncls)

trainDf['sentence'] = trainDf["sentence"].str.lower()
trainDf['sentence'] = trainDf['sentence'].str.replace('[^\w\s]','')
trainDf['sentence'] = trainDf["sentence"].fillna("fillna")

#%% Tokeninzing

MaxWords = 10000   
MaxLen = 400        
Tkns = tf.keras.preprocessing.text.Tokenizer(num_words = MaxWords)
SntncList = list(trainDf['sentence'])
Tkns.fit_on_texts(SntncList)
#%% convert tokens to sequce of numbers
# print(Tkns.word_index)
VocSize = len(Tkns.word_index) + 1
trainData = Tkns.texts_to_sequences(list(trainDf['sentence']))
# padding seuqnces
trainData = tf.keras.preprocessing.sequence.pad_sequences(trainData,
                                                       maxlen=MaxLen,
                                                       padding='pre')
#%% Train Test Split
XTrain,XTest,YTrain,YTest = train_test_split(trainData,target,test_size=0.2,
                                                            random_state=42)
#%% creat model
WVD = 100
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=VocSize,output_dim=WVD,
                          input_length=MaxLen))
model.add(tf.keras.layers.Flatten())                                 
model.add(tf.keras.layers.Dense(Ncls,activation='softmax'))
#%% compile model
model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])
#%% summary
model.summary()
#%% Training
NEpochs = 3
History = model.fit(XTrain,YTrain,epochs=NEpochs,validation_data=(XTest,YTest),
                    shuffle=True)

#%% Plot
TLoss = History.history['loss']
TAcc = History.history['accuracy']
VLoss = History.history['val_loss']
VAcc = History.history['val_accuracy']
epochs = History.epoch
fig1 = plt.figure(1)
p1 = plt.plot(epochs,TLoss)
p2 = plt.plot(epochs,VLoss)
plt.legend(['Train','Validation'])
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.savefig('Loss')

fig2 = plt.figure(2)
plt.plot(epochs,TAcc)
plt.plot(epochs,VAcc)
plt.legend(['Train','Validation'])
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.savefig('Accuracy')
#%% save model
model.save('LangDetection')
#%% Prediction Functio
def predict(txt):
    lang=['English','French','Italian','Spanish','Persian']
    txt = Tkns.texts_to_sequences([txt])
    txt = tf.keras.preprocessing.sequence.pad_sequences(txt,maxlen=MaxLen)
    prd = model.predict(txt)
    prd = prd.argmax() 
    return lang[prd]
    
