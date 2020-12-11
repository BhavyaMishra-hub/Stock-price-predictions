# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 22:13:10 2020

@author: DELL
"""


### Took Bhavya's model_features csv

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
import pandas as pd
import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


#Change working directory
import os 
os.chdir(r"D:\GitHub\Stock-price-predictions")  


data = pd.read_csv('model_features.csv')

#Creating HighLow and CloseOpen
data['HighLow'] = data['High'] - data['Low']
data['CloseOpen'] = data['Close'] - data['Open']

data = data.drop('Unnamed: 0', axis = 1)

#Creating Binary Classifier Momemtum
date = data['Date']
Open = data['Open']
Close = data['Close']
Momentum = [0]* len(date)

for i in range (1,len(date)):
    if Close[i] > Close[i-1] :
        Momentum[i] = "1"
    else :
        Momentum[i] = "0"
        
        
momen = pd.DataFrame(Momentum)
data = data.merge(momen, left_index = True, right_index = True)
data = data.rename({0: 'Momentum'}, axis = 1)


data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Month-str-full'] = data['Date'].dt.strftime('%B')        

# I keep 2015 - 16 as test and 2008 - 2014 as train

train_pd = data[data['Year'].isin(['2015', '2016'])]
test_pd = data[data['Year'].isin(['2008', '2009', '2010', '2011', '2012', '2013', '2014'])]

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'News Sentiment']



X_train = np.array(train_pd[features], dtype = 'float')
y_train = np.array(train_pd['Momentum'], dtype = 'float')
X_test = np.array(test_pd[features], dtype = 'float')
y_test = np.array(test_pd['Momentum'], dtype = 'float')


#Using SVM 


model = SVC()
model.fit(X_train, y_train)
ypred = model.predict(X_test)
accuracy_score(y_test, ypred)
