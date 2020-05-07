# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


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

FUTURE_PERIOD_PREDICT = 2



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

for x in range(20):
    data = pd.read_csv(r"../input/rttdata/ExpID_1_df_125_15min_originalRTT.csv") 
    
    #create new column, might need old one later? 
    data['future'] = data[f"RTT"].shift(-FUTURE_PERIOD_PREDICT+1)

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

#x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 4)



    
    #WHEN STACKING LSTM LAYERS WE MUST SET RETURN SEUQNCE = TRUE BECAUSE THE NEXT LSTM LAYER EXPECTS A SEQUENCE.
    model = Sequential()
    model.add(LSTM(64,input_shape=(x_train.shape[1:]),activation="tanh",return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    #model.add(LSTM(64,input_shape=(x_train.shape[1:]),activation="tanh",return_sequences=True))
    #model.add(Dropout(0.1))
    #model.add(BatchNormalization())

    model.add(LSTM(64,activation="relu",input_shape=(x_train.shape[1:])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(16,activation="relu"))
    model.add(Dropout(0.2))


    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')


    history = model.fit(x_train, y_train, epochs=50, batch_size=40, validation_data=(x_test, y_test), verbose=1, shuffle=False)



    # plot history
    #pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    #pyplot.legend()
    #pyplot.show()

    model.save("/kaggle/working/4layern="+ str(FUTURE_PERIOD_PREDICT) +  ".h5")

    import pickle

    k = open("/kaggle/working/4layerHistoryn="+str(FUTURE_PERIOD_PREDICT) + ".pckl", 'wb')
    pickle.dump(history.history, k)
    k.close()
    
    FUTURE_PERIOD_PREDICT = FUTURE_PERIOD_PREDICT + 1

    #y_test = y_test.to_numpy()
    #y_test = y_test.reshape(len(y_test),1)
    #y_train = y_train.to_numpy()
    #y_train = y_train.reshape(len(y_train),1)

    #trainPredic = model.predict(x_train)

    #testPredic = model.predict(x_test)

    #print(NMAE(y_train,trainPredic))
    #print(NMAE(y_test,testPredic))