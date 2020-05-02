# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:30:37 2020

@author: wietz
"""

import pandas
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Dropout,LSTM,BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
from collections import deque
from sklearn.model_selection import train_test_split

FUTURE_PERIOD_PREDICT = 10



data = pandas.read_csv(r"C:\Users\wietz\Desktop\IoT\ExpID_1_df_125_15min_originalRTT.csv") 

#Preporcess

def NMAE(y,yHat):
    return np.sum(np.absolute(y - yHat)) / np.sum(y)


def preprocessDf(df):
    
    tmp = df["logTS"].to_numpy()
    for x in range(len(tmp)):
        tmp[x] = tmp[x][0:2]
        
    df["hour"] = tmp
    
        
    #TODO should be two classes, weekend or weekday
    tmp = df["logDate"].to_numpy()
    for x in range(len(tmp)):
        tmp[x] = tmp[x][8:10]    
    
    df["day"] = tmp
    
    #columns that are not necessary should be dropped.
    df = df.drop(columns=['srcIP', 'Loss','Timestamp', 'DateTime','logDate','logTS',"date","time","RTT"])
    
    y = df.pop("future")


#    #normalize data
#    for col in df.columns:
#        #normalzie everything except target?
#        if col != "target":
#            #df[col] = preprocessing.scale(df[col].values)
            
    # in case scale generates na
    df.dropna(inplace=True)
    
    sequential_data =[]
     
    #deque will pop values when it reaches nFeatures
    prevTimeSteps = deque(maxlen=FUTURE_PERIOD_PREDICT)
    
    for i in df.values:
        #-1 is for not taking target
        prevTimeSteps.append([float(n) for n in i[:]])
        if len(prevTimeSteps) == FUTURE_PERIOD_PREDICT:
            sequential_data.append(np.array(prevTimeSteps))
            
    #random.shuffle(sequential_data)
    
    #the data should be balanced.
    
    
    x = []
    
    
    for seq in sequential_data:
        x.append(seq)
        
    return np.array(x),y


#create new column, might need old one later? 
data['future'] = data[f"RTT"].shift(-FUTURE_PERIOD_PREDICT)
#to make it faster
data = data[:int(len(data)*0.05):]

x, y = preprocessDf(data)

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 4)



model = Sequential()
model.add(LSTM(128,input_shape=(train_x.shape[1:]),activation="tanh",return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape=(train_x.shape[1:]),activation="tanh",return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128,activation="relu",input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32,activation="relu"))
model.add(Dropout(0.2))