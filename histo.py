# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:08:00 2020

@author: wietz
"""

import pandas as pd
from matplotlib import pyplot
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pickle


FUTURE_PERIOD_PREDICT = 2

 


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
    y.dropna(inplace=True)


    #normalize data
    for col in df.columns:
        #normalzie everything except target?
        if col != "target":
            df[col] = preprocessing.scale(df[col].values)
            
    # in case scale generates na
    #df.dropna(inplace=True)
    
    sequential_data =[]
     
    #deque will pop values when it reaches nFeatures
    prevTimeSteps = deque(maxlen=FUTURE_PERIOD_PREDICT)
    
    for i in df.values[:-1]:
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





    
res = pd.DataFrame()

for i in range(20):
    FUTURE_PERIOD_PREDICT = 2 + i
    
    #create new column, might need old one later? 
    data = pd.read_csv(r"C:\Users\wietz\Desktop\IoT\ExpID_1_df_125_15min_originalRTT.csv")
    data['future'] = data["RTT"].shift(-FUTURE_PERIOD_PREDICT+1)
    

    #to make it faster
    #data = data[:int(len(data)*0.05):]
    
    x, y = preprocessDf(data)
    #drop last value, hack solution, try to fix this later
    y.drop(y.tail(1).index,inplace=True) 
    
    msk = np.random.rand(len(x)) < 0.8
    x_train = x[msk]
    x_test = x[~msk]
    
    y_train = y[msk]
    y_test = y[~msk]
    
    
    
    #f = open('EngL.pckl', 'rb')
    #history = pickle.load(f)
    #f.close()
    
    
    model = Sequential()
    model.add(LSTM(64,input_shape=(FUTURE_PERIOD_PREDICT,47),activation="tanh",return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    #model.add(LSTM(64,input_shape=(x_train.shape[1:]),activation="tanh",return_sequences=True))
    #model.add(Dropout(0.1))
    #model.add(BatchNormalization())
    
    model.add(LSTM(64,activation="relu",input_shape=(FUTURE_PERIOD_PREDICT ,47)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(16,activation="relu"))
    model.add(Dropout(0.2))
    
    
    model.add(Dense(1))
    
    model.compile(loss='mae', optimizer='adam')
    
    
    checkpoint_path = r"C:\Users\wietz\Desktop\4layern="+ str(FUTURE_PERIOD_PREDICT) + ".h5"
    
    model.load_weights(checkpoint_path)
    
    
    y_test = y_test.to_numpy()
    y_test = y_test.reshape(len(y_test),1)
    y_train = y_train.to_numpy()
    y_train = y_train.reshape(len(y_train),1)
    
    trainPredic = model.predict(x_train)
    
    testPredic = model.predict(x_test)
    
    
    res = res.append({"n":2+i,
                      "nmaeTest":NMAE(y_test,testPredic),
                      "nmaeTrain" :NMAE(y_train,trainPredic)}, ignore_index=True)
    print(i)
    
