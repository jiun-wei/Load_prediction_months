
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import configparser
from datetime import datetime,timedelta
import calendar
from dateutil.relativedelta import relativedelta 
from pandas import read_csv
import csv
import pickle as pkl
import os

os.environ["CUDA_VISIBLE_DEVICES"] ="1" #指定使用編號1的GPU



series= read_csv('2018-avg-test-month.csv')

time=series['DateTime']
time_last=time[-1:]
del series['DateTime']
time_last=np.array(time_last)
total_time=[]


config = configparser.ConfigParser() #Load ini 檔
config.read('Setting.ini')


Model1= (config.get('Set','model1'))
Model2= (config.get('Set','model2'))
Model3= (config.get('Set','model3'))


timeseries=series.values

power=timeseries[:,-1]
timeseries=timeseries[:,:-1]
with open("save/avgpowermonthallnew-"+Model1+".pkl","rb") as infile: #load正規化的weight
    ascalermax0_1 = pkl.load(infile)
timeseries= ascalermax0_1.transform(timeseries)
realvalue=np.array(power).reshape(power.shape[0],1)
model=load_model("save/avgpowermonthallnew-"+Model1+".h5")

pred1 = model.predict(timeseries)
pred1 = pred1.astype(np.float32)
pred1=np.array(pred1).reshape(power.shape[0],1)
with open("save/avgpowernew-"+Model1+".pkl","rb") as infile: #load正規化的weight
    ascalermax0_1 = pkl.load(infile)

pred1=ascalermax0_1.inverse_transform(pred1)

print('pred1',pred1)
pred1=np.array(pred1).reshape(pred1.shape[0],1)
pred1=pred1.astype('float32')



timeseries=series.values

power=timeseries[:,-1]
timeseries=timeseries[:,:-1]
with open("save/avgpowermonthallnew-"+Model2+".pkl","rb") as infile: #load正規化的weight
    ascalermax0_1 = pkl.load(infile)
timeseries= ascalermax0_1.transform(timeseries)
realvalue=np.array(power).reshape(power.shape[0],1)
model=load_model("save/avgpowermonthallnew-"+Model2+".h5")

pred2 = model.predict(timeseries)
pred2 = pred2.astype(np.float32)
pred2 = np.array(pred2).reshape(power.shape[0],1)
with open("save/avgpowernew-"+Model2+".pkl","rb") as infile: #load正規化的weight
    ascalermax0_1 = pkl.load(infile)

pred2=ascalermax0_1.inverse_transform(pred2)

print('pred2',pred2)
pred2=np.array(pred2).reshape(pred2.shape[0],1)
pred2=pred2.astype('float32')





timeseries=series.values

power=timeseries[:,-1]
timeseries=timeseries[:,:-1]
with open("save/avgpowermonthallnew-"+Model3+".pkl","rb") as infile: #load正規化的weight
    ascalermax0_1 = pkl.load(infile)
timeseries= ascalermax0_1.transform(timeseries)
realvalue=np.array(power).reshape(power.shape[0],1)
model=load_model("save/avgpowermonthallnew-"+Model3+".h5")

pred3 = model.predict(timeseries)
pred3 = pred3.astype(np.float32)
pred3 = np.array(pred3).reshape(power.shape[0],1)
with open("save/avgpowernew-"+Model3+".pkl","rb") as infile: #load正規化的weight
    ascalermax0_1 = pkl.load(infile)

pred3 = ascalermax0_1.inverse_transform(pred3)

print('pred3',pred3)
pred3 = np.array(pred3).reshape(pred3.shape[0],1)
pred3 = pred3.astype('float32')


pred=[]


for i in range(power.shape[0]):
    pred.append((pred1[i]+pred2[i]+pred3[i])/3)

pac_list=[]
pac_list=abs((pred-realvalue)/realvalue)*100



error=(abs(pred-realvalue))
    
avg_error=0
for x in range(12):
    avg_error+=(error[x]/12)
print('avg_error',avg_error)

month = [1,2,3,4,5,6,7,8,9,10,11,12]

#month = [1,2,3,4,5,6,7,8]

# total_time=np.array(total_time)
# total_time=total_time.reshape(12,1)
# pred_all=np.hstack((total_time,pred))
   

# with open('PredictavgmonthOutput.csv','w',newline ='',encoding ='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow(['DateTime','Power'])
#         writer.writerows(pred_all)


plt.figure() #輸出圖形
plt.plot(month,pac_list,'blue', linewidth=3,label='MAPE')
plt.ylim(0,4)
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
plt.xticks(month)
plt.legend(prop=font1)
plt.tick_params(labelsize=20) 


plt.figure()
plt.plot(month,pred,'red', linewidth=3,label='Predicted')
plt.plot(month,realvalue,'green',linewidth=3, label='Actual') 
plt.ylim(75000,84000)
plt.xticks(month)   
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }
plt.legend(prop=font1)
plt.tick_params(labelsize=20)

plt.figure()
plt.plot(month,error,'blue',linewidth=3,label='MAE')
plt.xticks(month)
plt.ylim(0,3500)
plt.legend(prop=font1)
plt.tick_params(labelsize=20)
plt.show() 
