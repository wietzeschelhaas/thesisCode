# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:55:11 2020

@author: wietze
"""

import pandas
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing



# load dataset

data = pandas.read_csv(r"C:\Users\wietz\Desktop\IoT\ExpID_1_df_125_15min_originalRTT.csv") 
#data = pandas.read_csv(r"/kaggle/input/ExpID_1_df_125_15min_originalRTT.csv")


#rtt is target, 
Y = data.pop("RTT")


tmp = data["logTS"].to_numpy()
for x in range(len(tmp)):
    tmp[x] = tmp[x][0:2]
    
data["hour"] = tmp

    
tmp = data["logDate"].to_numpy()
for x in range(len(tmp)):
    tmp[x] = tmp[x][8:10]    

data["day"] = tmp

tmp = data["Timestamp"].to_numpy()
for x in range(len(tmp)):
    tmp[x] = tmp[x][14:16]    

data["minute"] = tmp


#columns that are not necessary should be dropped.
data = data.drop(columns=['srcIP', 'Loss','Timestamp', 'DateTime','logDate','logTS',"date","time"])
 
#get certain collumns. exluding srcID since its the same???
#basic = data[['src_x','src_y','node_x','node_y','distance','NodeID','CH','RSSI','LQI']]

#normalize data:
x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = pandas.DataFrame(x_scaled)


#basic = pandas.DataFrame()
#basic['a'] = np.asarray(range(1000))
#basic['b'] = np.asarray(range(1000,2000))
#
#
#Y = np.asarray(range(4000,5000))


x_train, x_test,y_train,y_test = train_test_split(data,Y,test_size = 0.2, random_state = 4)



# define base model
#mean squared error is optimized. 
#no activation funciton is used on the output since this is regression.
 
def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(48, input_dim=48, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

    

estimator = KerasRegressor(build_fn=baseline_model, epochs=1, batch_size=5, verbose=1)

estimator.fit(x_train, y_train)

predicTest = estimator.predict(x_test)
predicTrain = estimator.predict(x_train)

#kfold = KFold(n_splits=5)
#results = cross_val_score(estimator, data, Y, cv=kfold)
#print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))




def NMAE(y,yHat):
    return np.sum(np.absolute(y - yHat)) / np.sum(y)


print(NMAE(y_test,predicTest))
print(NMAE(y_train,predicTrain))



#print(NMAE(y_test,predicTest))
#print(NMAE(y_train,predicTrain))





