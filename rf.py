# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:27:12 2020

@author: wietz
"""

import pandas
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
import numpy as np


data = pandas.read_csv(r"C:\Users\wietz\Desktop\IoT\ExpID_1_df_125_15min_originalRTT.csv") 

 

data['rank'] = 0.0
data['nbr'] = 0.0

for k in range(len(data)):

    nodeid_str = data['NodeID'][k].astype(int).astype(str)
    
    data.loc[k, 'rank'] = data['rank_' + nodeid_str][k]
    
    data.loc[k, 'nbr'] = data['nbr_' + nodeid_str][k]


iot_features_all = ['src_x', 'src_y', 'node_x', 'node_y', 'distance', 'time', 'date', 'srcIP', 'NodeID', 'CH', 'LQI', 'RSSI',

            'rank_121', 'rank_122', 'rank_123', 'rank_124', 'rank_125', 'rank_126', 'rank_128', 'rank_129', 'rank_130', 'rank_131', 'rank_132', 'rank_134', 'rank_135', 'rank_136', 'rank_137', 'rank_138', 'rank_139', 'rank_140',

            'nbr_121', 'nbr_122', 'nbr_123', 'nbr_124', 'nbr_125', 'nbr_126', 'nbr_128', 'nbr_129', 'nbr_130', 'nbr_131', 'nbr_132', 'nbr_134', 'nbr_135', 'nbr_136', 'nbr_137', 'nbr_138', 'nbr_139', 'nbr_140',

            'Light']

iot_features = ['src_x', 'src_y', 'node_x', 'node_y', 'distance', 'time', 'date', 'CH', 'LQI', 'RSSI', 'rank', 'nbr', 'Light']

 

X = data[iot_features]

y = data['RTT']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)


pred_method = 'RF regr'



# these hyper params are just a guess
regr = RandomForestRegressor(n_estimators=1000, max_depth=20, random_state=42)

regr.fit(X_train, y_train)

# Predict numerical y values

y_num_pred_RFregr_train = regr.predict(X_train)

y_num_pred_RFregr_test = regr.predict(X_test)

def NMAE(y,yHat):
    return np.sum(np.absolute(y - yHat)) / np.sum(y)
    