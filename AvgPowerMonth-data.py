
import pandas as pd
import numpy as np
from keras.models import load_model
import configparser
from datetime import datetime,timedelta
import calendar
from dateutil.relativedelta import relativedelta 
from pandas import read_csv
import csv
import pickle as pkl
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] ="1" #指定使用編號1的GPU



series= read_csv('TrainAvgMonth-2018.csv')
data=series['HL6B'].values


def average(data):
    return np.sum(data)/len(data)
#标准差
def sigma(data,avg):
    sigma_squ=np.sum(np.power((data-avg),2))/len(data)
    return np.power(sigma_squ,0.5)
#高斯分布概率
def prob(data,avg,sig):
    print(data)
    sqrt_2pi=np.power(2*np.pi,0.5)
    coef=1/(sqrt_2pi*sig)
    powercoef=-1/(2*np.power(sig,2))
    mypow=powercoef*(np.power((data-avg),2))
    return coef*(np.exp(mypow))
#样本数据
# data=np.array([0.79,0.78,0.8,0.79,0.77,0.81,0.74,0.85,0.8
#                ,0.77,0.81,0.85,0.85,0.83,0.83,0.8,0.83,0.71,0.76,0.8])
#根据样本数据求高斯分布的平均数
ave=average(data)
#根据样本求高斯分布的标准差
sig=sigma(data,ave)
#拿到数据
x=np.arange(min(data),max(data),50)

p=prob(x,ave,sig)
plt.plot(x,p, linewidth=3)
plt.grid()
# plt.xlabel("HLB")
# plt.ylabel("prob density")
# plt.yticks(np.arange(0,12,1))
plt.title("Gaussian distrbution")


plt.tick_params(labelsize=20) 
plt.show()






# plt.figure() #輸出圖形
# plt.plot(day,pac_list,'blue', linewidth=3,label='MAPE')
# plt.ylim(0,4)
# font1 = {'family': 'Times New Roman',
#           'weight': 'normal',
#           'size': 20,
#           }
# plt.xticks(day)
# plt.legend(prop=font1)
# plt.tick_params(labelsize=20) 
    

# plt.figure()
# plt.plot(day,pred,'red', linewidth=3,label='Predicted')
# plt.plot(day,realvalue,'green',linewidth=3, label='Actual')
# plt.ylim(73000,81000)    
# font1 = {'family': 'Times New Roman',
#           'weight': 'normal',
#           'size': 20,
#           }
# plt.xticks(day)
# plt.legend(prop=font1)
# plt.tick_params(labelsize=20)
    
# plt.figure()
# plt.plot(day,error,'blue',linewidth=3,label='MAE')
# plt.ylim(0,3000)  
# plt.legend(prop=font1)
# plt.tick_params(labelsize=20)
# plt.show() 






