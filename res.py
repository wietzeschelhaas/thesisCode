# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:30:09 2020

@author: wietz
"""

import pandas
from matplotlib import pyplot
import numpy as np

layer4Units16 = pandas.read_csv(r"C:\Users\wietz\Google Drive\thesis\results\4layer16Units.csv") 
layer4Units32 = pandas.read_csv(r"C:\Users\wietz\Google Drive\thesis\results\4layer32Units.csv") 
layer4Units64 = pandas.read_csv(r"C:\Users\wietz\Google Drive\thesis\results\4layer64Units.csv") 
layer5Units64 = pandas.read_csv(r"C:\Users\wietz\Google Drive\thesis\results\5layer64Units.csv") 
layer5Units32 = pandas.read_csv(r"C:\Users\wietz\Google Drive\thesis\results\4layer32Units.csv") 


fig, axs = pyplot.subplots(2,3)
fig.tight_layout(pad=2)

axs[0,0].plot(layer4Units16["n"],layer4Units16['nmaeTest'], label='NMAE test')
axs[0,0].plot(layer4Units16["n"],layer4Units16['nmaeTrain'], label='NMASE train')

miny = min(layer4Units16["nmaeTest"])
minx = layer4Units16["n"].to_numpy()[np.where(layer4Units16["nmaeTest"] == miny)]

axs[0, 0].set_title('4 layers, 16 units,\n lowest test NMAE = ' + str(miny) + "\n n = " + str(minx))

axs[0,1].plot(layer4Units32["n"],layer4Units32['nmaeTest'], label='NMAE test')
axs[0,1].plot(layer4Units32["n"],layer4Units32['nmaeTrain'], label='NMASE train')

miny = min(layer4Units32["nmaeTest"])
minx = layer4Units32["n"].to_numpy()[np.where(layer4Units32["nmaeTest"] == miny)]

axs[0, 1].set_title('4 layers, 32 units,\n lowest test NMAE = ' + str(miny) + "\n n = " + str(minx))

axs[0,2].plot(layer4Units64["n"],layer4Units64['nmaeTest'], label='NMAE test')
axs[0,2].plot(layer4Units64["n"],layer4Units64['nmaeTrain'], label='NMASE train')

miny = min(layer4Units64["nmaeTest"])
minx = layer4Units64["n"].to_numpy()[np.where(layer4Units64["nmaeTest"] == miny)]

axs[0, 2].set_title('4 layers, 64 units,\n lowest test NMAE = ' + str(miny) + "\n n = " + str(minx))

axs[1,0].plot(layer5Units32["n"],layer5Units32['nmaeTest'], label='NMAE test')
axs[1,0].plot(layer5Units32["n"],layer5Units32['nmaeTrain'], label='NMASE train')

miny = min(layer5Units32["nmaeTest"])
minx = layer5Units32["n"].to_numpy()[np.where(layer5Units32["nmaeTest"] == miny)]

axs[1, 0].set_title('5 layers, 32 units,\n lowest test NMAE = ' + str(miny) + "\n n = " + str(minx))

axs[1,1].plot(layer5Units64["n"],layer5Units64['nmaeTest'], label='NMAE test')
axs[1,1].plot(layer5Units64["n"],layer5Units64['nmaeTrain'], label='NMASE train')

miny = min(layer5Units64["nmaeTest"])
minx = layer5Units64["n"].to_numpy()[np.where(layer5Units64["nmaeTest"] == miny)]

axs[1, 1].set_title('5 layers, 64 units,\n lowest test NMAE = ' + str(miny) + "\n n = " + str(minx))

pyplot.show()