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
path = os.getcwd()


sentiment_df = pd.read_excel("news_sentiment_data.xlsx",sheet_name='Data')
prices_df = pd.read_csv("DJIA Index.csv")
sentiment_df['date']= pd.to_datetime(sentiment_df['date'])
prices_df['Date']= pd.to_datetime(prices_df['Date'])

data = prices_df.merge(sentiment_df, left_on = 'Date', right_on = 'date')



#model_features.to_csv("model_features.csv")

data = data.drop('date', axis = 1)

#Creating HighLow and CloseOpen
data['HighLow'] = data['High'] - data['Low']
data['CloseOpen'] = data['Close'] - data['Open']



#Creating Binary Classifier Momemtum
date = data['Date']
Open = data['Open']
Close = data['Close']
Momentum = [0]* len(date)

for i in range (1,len(date)):
    if Close[i] > Close[i-1] :
        Momentum[i] = "1"
    else :
        Momentum[i] = "-1"
        
        
        
momen = pd.DataFrame(Momentum)
data = data.merge(momen, left_index = True, right_index = True)
data = data.rename({0: 'Momentum'}, axis = 1)

#Giving a lag to momentum
data['Momentum'] = data['Momentum'].shift(-1)


#Removing last row
data.drop(data.tail(1).index,inplace=True)



data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Month-str-full'] = data['Date'].dt.strftime('%B')        

# I keep 2015 - 16 as test and 2008 - 2014 as train

test_pd = data[data['Year'].isin(['2015', '2016'])]
train_pd = data[data['Year'].isin(['2008', '2009', '2010', '2011', '2012', '2013', '2014'])]

features = ['Open', 'High', 'Low', 'Volume', 'News Sentiment']



X_train = np.array(train_pd[features], dtype = 'float')
y_train = np.array(train_pd['Momentum'], dtype = 'float')
X_test = np.array(test_pd[features], dtype = 'float')
y_test = np.array(test_pd['Momentum'], dtype = 'float')


#Using SVM 
model1 = SVC()
model1.fit(X_train, y_train)
ypred = model1.predict(X_test)
accuracy_score(y_test, ypred)

error_SVM1 = 1 - (sum(np.where(ypred>0, 1, -1) == y_test))/368
#this error can be compared with the error that other OLS is giving



#Xtest = pd.DataFrame(Xtest1)
    #Ytest = pd.DataFrame(Ytest1)
#    yhat = Xtest@w
#    y_calc_sign = np.sign(yhat)
#    count = 0
#    for i in range(len(Ytest)):
#          if(Ytest.iloc[i] != y_calc_sign.iloc[i]):
#            count +=1
    

#Using Ridge Regression
def error_func(w, Xtest, Ytest):
    error_rates = []
    
    error_rate = sum(np.where(Xtest@w>0, 1, -1) == Ytest)
    error_rates.append(float(368-error_rate)/368)
    
    
    return error_rates


#Ridge function takes the training and test datasets and the D/I matrix
#For a range of lambdas it checks which give the least error and returns 
#the weights that give the minimum error
    
def ridgeX(x_train, y_train, x_test, y_test, matrix): 
       
    min_w = np.linalg.inv(x_train.T@x_train + 5*matrix)@x_train.T@y_train
    #Here the min_w is calculated with lambda = 5

    min_error = error_func(min_w, x_test, y_test)    
    #print(min_error.shape)

    for i in [5, 10, 15, 50]:
        w = np.linalg.inv(x_train.T@x_train + i*matrix.T@matrix)@x_train.T@y_train
        #print(w.shape)

        error = error_func(w, x_test, y_test)
        #print(error.shape)

        if((sum(error)/len(error))<(sum(min_error)/len(error))):
            min_w = w
            min_error = error
        #This checks if the errors of each lambda are lesser than the errors when lambda = 1. If yes then it gives 
        # weights corresponding to that lambda    
            
        #The function however does not tell which lambda gave the least error
        # ie. we dont know which is best lambda.     

    return min_w


# Making D Matrix
n,m = X_train.shape
D = np.zeros((m,m))

D[0,0] = 1
D[1,0] = -2
D[1,1] = 1

for i in range(2,5):
    D[i,i] = 1
    D[i, i-1] = -2
    D[i, i-2] = -1
    
    
# Making Identity matrix    

I = np.identity(m)    
    


w_ridgeD = ridgeX(X_train, y_train, X_test, y_test, D)
w_ridgeI = ridgeX(X_train, y_train, X_test, y_test, I)
w_ols = la.inv(X_train.T@X_train)@X_train.T@y_train


error_ridgeD = error_func(w_ridgeD, X_test, y_test)
error_ridgeI = error_func(w_ridgeI, X_test, y_test)
error_w_ols = error_func(w_ols, X_test, y_test)

## Ridge Regression with D matrix and OLS always give the same weights.



#################################################################
# https://towardsdatascience.com/implement-svm-with-python-in-2-minutes-c4deb9650a02

## Try SVM without Scikit-Learn
def loss(W, x, y, C):
    return 1/2 * np.sum(W**2) + C * np.sum([np.max([0, 1 - y[i] * (W @ x[i])]) for i in range(x.shape[0])])


def lossGradient(W, x, y, C):
    '''
    Loss gradient for SGD with batch size 1 only
    '''
    lossGrad = np.zeros_like(W)
    distance = np.max([0, 1 - y * (W @ x)])
    if distance == 0:
        lossGrad = W
    else:
        lossGrad = W - C * y * x
            
    return lossGrad


W = np.random.random(X_train.shape[1])
N_STEPS = 100
lr = 1e-4
C = 1e-4

for step in range(N_STEPS):
    for pi, p in enumerate(X_train):
        W = W - lr*lossGradient(W, p, y_train[i], C = C)
    if step % 10 == 0:
        print(f'Current Loss {loss(W, X_train, y_train,C = C)} Step {step}')
        
w_SVM = W

error_SVM2 = error_func(W, X_test, y_test)

## The final error rates are error_ridgeD, error_ridgeI, error_w_ols, error_SVM
error_ridgeD
error_ridgeI
error_w_ols
error_SVM1 #same
error_SVM2 #same



    
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Weights from different columns')
ax1.plot(w_ols)
ax1.set_title('OLS')
ax2.plot(w_SVM, 'tab:orange')
ax2.set_title('SVM')
ax3.plot(w_ridgeD, 'tab:green')
ax3.set_title('Ridge Reg (D)')
ax4.plot(w_ridgeI, 'tab:red')
ax4.set_title('Ridge Reg (I)')
for ax in fig.get_axes():
    ax.label_outer()
    
fig.savefig(os.path.join(path))    
