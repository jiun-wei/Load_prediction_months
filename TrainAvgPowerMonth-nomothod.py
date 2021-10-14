# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:33:20 2020

@author: JimTseng
"""
from __future__ import print_function, division
import numpy as np
from keras.layers import Convolution1D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
import random
from pandas import read_csv
from pandas import DataFrame
from random import sample
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
from keras.models import load_model
import configparser
import re
from copy import deepcopy
from sklearn.model_selection import train_test_split
import os

def new_data(data,n=4):
    Hcol = [col for col in data.columns if re.findall('H(L[0-9][A-Z]_(Array|CELL)|C[0-9][A-Z])_Input',col)]
    newdata = deepcopy(data)
    for k in range(n):
        data1 = deepcopy(data)
        data1[Hcol] = data1[Hcol].values + [[random.uniform(0,10) for i in range(len(Hcol))]for j in range(data1.shape[0])]    
        newdata = pd.concat([newdata,data1], ignore_index = True)
    return newdata


def dataProcess_shuffle(timeseries, window_size, n_steps_out):
     data_container = []
     x_data = []
     y_data = []
     for i in range(0,len(timeseries) - (window_size + n_steps_out) + 1,window_size+n_steps_out):
         if i == 0:
             data_container = [timeseries[0:(window_size + n_steps_out)]]
         else:
             data_container.append(timeseries[i:(i + window_size + n_steps_out)])
     random.shuffle(data_container)
     for i in range(len(data_container)):
        if i == 0:
           x_data = [data_container[0][0:window_size]]
           y_data = [data_container[0][window_size:window_size + n_steps_out]]
        else:
            x_data.append(data_container[i][0:window_size])
            y_data.append(data_container[i][window_size:window_size + n_steps_out])

     np_x_data= np.array(x_data)
     np_y_data = np.array(y_data)
     return np_x_data, np_y_data


def make_timeseries_regressor(window_size,nb_input_series, n_steps_out):
    model = Sequential((
        Dense(12, input_shape=(window_size,), activation='relu'),
        Dense(n_steps_out, activation='linear'), 
    ))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


if __name__ == '__main__':
    

    
    
        
    dataset = read_csv('TrainAvgMonth-2018.csv',encoding='utf-8') #讀取資料
    time=dataset['DateTime']
    del dataset['DateTime']
    window_size, n_steps_out = dataset.shape[1]-1 , 1 # window_size為設定input_size, n_steps_out為設定output_size
   
    


    power=dataset['Energy'].values  
    power=power.reshape(power.shape[0],1)
    del dataset['Energy']
    new_timeseries=dataset.values
    ascalermax0_1 = MinMaxScaler(feature_range=(0,1))#將所有資料做正規化(0,1)
    new_timeseries = ascalermax0_1.fit_transform(new_timeseries)
    with open("models-2018/avgpowermonth-nomothod.pkl", "wb") as outfile: #將正規化的weight存成pkl
        pkl.dump(ascalermax0_1, outfile)
    ascalermax0_1 = MinMaxScaler(feature_range=(0,1))#將所有資料做正規化(0,1)
    power = ascalermax0_1.fit_transform(power)
    with open("models-2018/avgpower-nomothod.pkl", "wb") as outfile: #將正規化的weight存成pkl
        pkl.dump(ascalermax0_1, outfile)
    new_timeseries=np.hstack((new_timeseries,power))
    new_timeseries=new_timeseries.reshape(new_timeseries.shape[0]*new_timeseries.shape[1],1)
    train_X, train_y = dataProcess_shuffle(new_timeseries, window_size, n_steps_out)
    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1])
    train_y = train_y.reshape((train_y.shape[0],1))   
    model = make_timeseries_regressor(window_size=window_size, nb_input_series=1, n_steps_out=1)
    print('\n\nModel with input size {}, output size {}'.format(model.input_shape, model.output_shape))
    model.summary()   
    filename = ("models-2018/avgpowermonth-nomothod.h5").format()
    checkpoint = ModelCheckpoint(filename, monitor='mean_absolute_error', verbose=2,save_best_only = True, mode='min') #儲存訓練的最小誤差
    callbacks_list= [checkpoint]    
    history = model.fit(train_X, train_y, nb_epoch=5000, batch_size= 2,callbacks=callbacks_list, validation_split=0.05)



        
        
        
        
