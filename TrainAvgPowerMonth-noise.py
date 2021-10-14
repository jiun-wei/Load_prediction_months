from __future__ import print_function, division
import numpy as np
from keras.layers import Convolution1D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
import random
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
from keras.models import load_model
import heapq
import configparser
import csv 

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
   


def new_random(timeseries):
    temp=[]
    L6B=[]
    HL6B=[]
    Energy=[]
    Htemp=[]
#    SL6B=[]
#    Hum=[]
#    HHum=[]
    for i in range(timeseries.shape[1]):
        for j in range(timeseries.shape[0]):
             if i==0:
                a=timeseries[j][i]
                temp.append(a)# temp
             elif i==1:
                a=timeseries[j][i]
                Htemp.append(a)# HTemp
             elif i==2:
                a=timeseries[j][i]
                L6B.append(a)# L6B
             elif i==3:
                a=timeseries[j][i]+(random.uniform(-21.5,21.5))
                HL6B.append(a)# HL6B
#             elif i==4:
#                a=timeseries[j][i]
#                SL6B.append(a)# HL6B
#             elif i==4:
#                a=timeseries[j][i]
#                Hum.append(a)# Hum
#             elif i==5:
#                a=timeseries[j][i]
#                HHum.append(a)# HHum
             elif i==4:
                a=timeseries[j][i]
                Energy.append(a)# Energy
                


    np_new_temp= np.array(temp)
    np_new_Htemp= np.array(Htemp)
    np_new_L6B= np.array(L6B)
    np_new_HL6B= np.array(HL6B)
#    np_new_SL6B= np.array(SL6B)
#    np_new_Hum= np.array(Hum)
#    np_new_HHum= np.array(HHum)
    np_new_Energy= np.array(Energy)

    return np_new_temp,np_new_Htemp,np_new_L6B,np_new_HL6B,np_new_Energy

def new_data(timeseries):
    names = locals()
    for i in range(1,5): #生成data
                names['Temperature%s'%i],names['Htemperature%s'%i],names['L6B_Array_Input%s'%i],names['HL6B_Array_Input%s'%i],names['Energy%s'%i]=new_random(timeseries)
                names['totoal_data%s'%i]=np.vstack((names['Temperature%s'%i],names['Htemperature%s'%i],names['L6B_Array_Input%s'%i],names['HL6B_Array_Input%s'%i],names['Energy%s'%i]))
                names['totoal_data%s'%i] =  names['totoal_data%s'%i].T
    for i in  range(1,5):
        names['totoal_data%s'%i] = names['totoal_data%s'%i].astype('float32')
        timeseries=np.vstack((timeseries,names['totoal_data%s'%i])) #將所有資料統整(原本、生成)
    return timeseries

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
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
  

    return model



if __name__ == '__main__':


    dataset = read_csv('TrainAvgMonth-2018.csv',encoding='utf-8')
    time=dataset['DateTime']
    del dataset['DateTime']
    power=dataset['Energy']
    timeseries = dataset.values


    
    
    window_size, n_steps_out = 4 , 1 # window_size=input_size  

    reframed = series_to_supervised(timeseries,1, 1) #設置多元序列   
    reframed.drop(reframed.columns[[5,6,7,8,9]], axis=1, inplace=True)
    values = reframed.values
    

    new_timeseries=new_data(timeseries)
    power=new_timeseries[:,-1]   
    power=power.reshape(power.shape[0],1)
    new_timeseries=new_timeseries[:,:-1]
    
    ascalermax0_1 = MinMaxScaler(feature_range=(0,1))#將所有資料做正規化(0,1)
    new_timeseries = ascalermax0_1.fit_transform(new_timeseries)
    with open("models-2018/avgpowermonth-noise10.pkl", "wb") as outfile: #將正規化的weight存成pkl
        pkl.dump(ascalermax0_1, outfile)
    ascalermax0_1 = MinMaxScaler(feature_range=(0,1))#將所有資料做正規化(0,1)
    power = ascalermax0_1.fit_transform(power)
    with open("models-2018/avgpower-noise10.pkl", "wb") as outfile: #將正規化的weight存成pkl
        pkl.dump(ascalermax0_1, outfile)
    new_timeseries=np.hstack((new_timeseries,power))
    new_timeseries=new_timeseries.reshape(new_timeseries.shape[0]*new_timeseries.shape[1],1)
    train_X, train_y = dataProcess_shuffle(new_timeseries, window_size, n_steps_out)
    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1])
    train_y = train_y.reshape((train_y.shape[0],1))

    
    model = make_timeseries_regressor(window_size=window_size, nb_input_series=1, n_steps_out=1)
    print('\n\nModel with input size {}, output size {}'.format(model.input_shape, model.output_shape))
    model.summary()   
    
    filename = ("models-2018/avgpowermonth-noise10.h5").format()
    checkpoint = ModelCheckpoint(filename, monitor='mean_absolute_error', verbose=2,save_best_only = True, mode='min') #儲存訓練的最小誤差
    callbacks_list= [checkpoint]    
    history = model.fit(train_X, train_y, nb_epoch=5000, batch_size= 2,callbacks=callbacks_list, validation_split=0.05)
    model.save("models-2018/avgpowermonth-noise10.h5") #儲存 model
    # print('model and weights written to disk')



        
        
        
        
