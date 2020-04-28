# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:55:11 2020

@author: wietze
"""

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn import preprocessing



# load dataset

data = pandas.read_csv(r"C:\Users\wietz\Desktop\IoT\ExpID_1_df_125_15min_originalRTT.csv") 
#data = pandas.read_csv(r"/kaggle/input/ExpID_1_df_125_15min_originalRTT.csv")

#rtt is target, 
Y = data.pop("RTT")




#columns that are not necessary should be dropped.
data = data.drop(columns=['srcIP', 'Loss','Timestamp', 'DateTime','logDate' ,'logTS'])
 
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
	model.add(Dense(47, input_dim=47, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=1, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, data, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#
#model = Sequential()
#model.add(Dense(47, input_dim=47, kernel_initializer='normal', activation='relu'))
#model.add(Dense(1, kernel_initializer='normal'))
## Compile model
#model.compile(loss='mean_absolute_error', optimizer='adam')
#
#
#model.fit(x_train, y_train, validation_data=(x_test, y_test),
#	epochs=100, batch_size=8)



#target = data["RTT"]

#x_train, x_test,y_train,y_test = train_test_split(data,target,test_size = 0.2, random_state = 4)




