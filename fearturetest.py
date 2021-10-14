# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#c = [-10,-5,0,5,3,10,15,-20,25]
#
#print (c.index(min(c)))  # 返回最小值
#print(c.index(max(c))) # 返回最大值
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pandas import read_csv    
import numpy as np
import csv
import statsmodels.api as sm
# series= read_csv('TrainAvgMonth.csv')
# df = pd.DataFrame(series,columns=['Temperature','HighestTemp','L6B','HL6B','CELL','Energy'])
# df = pd.DataFrame(series,columns=['DateTime','Power','Temperature','Humidity'])
# corrMatrix = df.corr()

# sn.heatmap(corrMatrix, annot=True)
# plt.show()

# dta= read_csv('COMED_hourly_set.csv')

# sm.graphics.tsa.plot_pacf(dta[["COMED_MW"]].values.squeeze(), lags=24)

# plt.tick_params(labelsize=20)
# plt.show()





# dta= read_csv('TrainOneDay_2018.csv')

# sm.graphics.tsa.plot_pacf(dta[["Power"]].values.squeeze(), lags=96)

# plt.tick_params(labelsize=20)
# plt.show()


series= read_csv('2018-avg-train-month.csv')

df = pd.DataFrame(series,columns=['Energy','Temperature','HighestTemp','L6B','HL6B'])
corrMatrix = df.corr()
sn.set(font_scale=1.5)
sn.heatmap(corrMatrix, annot=True)
plt.tick_params(labelsize=15)
plt.show()