# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:46:57 2020

@author: wietz
"""

import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np

df = pd.read_csv(r"C:\Users\wietz\Desktop\IoT\crypto_data\BTC-USD.csv",names=["time","low","high","open","close","volume"])

main_df = pd.DataFrame() # begin empty

#look at last 60 minutes
SEQ_LEN = 60
#to predcit the coming 3 minutes
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocessDf(df):
    #dont need this anymore since it was only used to generate the target class
    df = df.drop("future",1)
    
    for col in df.columns:
        if col != "target":
            #computes the percentage change from the immediately previous row by default. 
            df[col] = df[col].pct_change()
            #drop the entire row if it contains na
            df.dropna(inplace=True)
            #normalize data, remember percent change can be above 1
            df[col] = preprocessing.scale(df[col].values)
    # in case scale generates na
    df.dropna(inplace=True)
    
    sequential_data =[]
    #deque will pop values when it reaches seq_len
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in df.values:
        #-1 is for not taking target
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
            
    random.shuffle(sequential_data)
    
    #the data should be balanced.
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        elif target == 1:
            buys.append([seq, target])


    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys),len(sells))
    
    #this balances the classes
    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys + sells
    
    random.shuffle(sequential_data)
    
    x = []
    y = []
    
    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)
        
    return np.array(x),y
 
ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider
for ratio in ratios:  # begin iteration
    print(ratio)
    dataset = 'C:\\Users\\wietz\\Desktop\\IoT\\crypto_data\\' + ratio + ".csv"
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

#every row is now gonna have a future column which is  the price in 3 minutes
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"],main_df["future"]))


#probaably already sorted, but its important that the validation data is in sequence so sort anyway.
times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

#validation data is the last 5 percent 
validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocessDf(main_df)
valid_x, valid_y = preprocessDf(validation_main_df)