import pandas as pd
from random import random
import numpy as np
import os
from math import exp

os.chdir("C:\\Users\\bhavy\\Documents\\GitHub\\Stock-price-predictions")

sentiment_df = pd.read_excel("news_sentiment_data.xlsx",sheet_name='Data')
prices_df = pd.read_csv("DJIA Index.csv")
sentiment_df['date']= pd.to_datetime(sentiment_df['date'])
prices_df['Date']= pd.to_datetime(prices_df['Date'])

model_features.to_csv("model_features.csv")

date = prices_df['Date']
Close = prices_df['Close']

Change = [0]* len(date)
Momentum = [0]* len(date)

for i in range (1,len(date)):
    if Close[i] > Close[i-1] :
        Momentum[i] = "1"
        Change[i] = (Close[i]-Close[i-1])/Close[i-1]
    else :
        Momentum[i] = "0"
        Change[i] = (Close[i-1] - Close[i]) / Close[i - 1]

model_features  = pd.DataFrame({'Date':date, 'Close':Close,'Change':Change,'Momentum':Momentum})
model_features = pd.merge(model_features,sentiment_df,left_on = 'Date',right_on = 'date')
model_features  = model_features.drop(columns = {'date'})
expected = model_features.loc[1:,'Close'].reset_index(drop = True)

def initialize_weights(n_inputs, n_hidden, n_outputs):
    print(n_hidden)
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

    def activate(weights, inputs):
        #print(weights)
        activation = weights[-1]
        #print(weights[-1])
        for i in range(len(weights) - 1):
            #print(inputs[i])
            activation += (weights[i] * inputs[i])
        return activation


    # Transfer neuron activation
    def transfer(activation):
        return 1.0 / (1.0 + exp(-activation))


    # Forward propagate input to a network output
    def forward_propagate(network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                #print(inputs)
                activation = activate(neuron['weights'], inputs)
                #print(activation)
                neuron['output'] = transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    def transfer_derivative(output):
        return output * (1.0 - output)


    # Backpropagate error and store in neurons
    def backward_propagate_error(network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


    # Update network weights with error
    def update_weights(network, row, l_rate):
        for i in range(len(network)):
            inputs = row
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += -1*l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += -1*l_rate * neuron['delta']


    # Train a network for a fixed number of epochs
    def train_network(network, train, expected, l_rate):

        for i in range(len(train)):
            #print(train[i])
            output = forward_propagate(network, train[i])
            error = (expected[i] - output) ** 2
            backward_propagate_error(network, expected[i])
            update_weights(network, train[i], l_rate)

    # Make a prediction with a network
    def predict(network, row):
        output = forward_propagate(network, row)
        return output


n_layer = 3
input_nodes = model_features.shape[1]-1
hidden_nodes  = 20
output_nodes = 1
learning_rate = 0.2
network = initialize_weights(input_nodes,hidden_nodes,output_nodes)
fold = 500
accuracy = []
increment = 100
while(fold<=1800):
    print(fold)

    train_df = model_features.iloc[:fold]
    train_df = train_df.drop(columns = {"Date"})
    train_df = np.asarray(train_df, dtype='float64')
    test_df = model_features.iloc[fold:fold+increment]
    test_df = test_df.drop(columns = {"Date"})
    test_df = np.asarray(test_df, dtype='float64')
    expected_df = expected[:fold]
    expected_test = expected[fold:fold+increment].reset_index(drop = True)
    train_network(network, train_df,expected_df,learning_rate)
    total_error = 0
    total = 0
    for i in range(len(test_df)):
        total = total+1
        prediction = predict(network, test_df[i])
        total_error = total_error + (expected_test[i] - prediction)

    accuracy.append(total_error/total)
    print(accuracy)
    fold = fold+increment
