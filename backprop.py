import pandas as pd
import random
import numpy as np
import os
from math import exp
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

os.chdir("C:\\Users\\bhavy\\Documents\\GitHub\\Stock-price-predictions")

sentiment_df = pd.read_excel("news_sentiment_data.xlsx",sheet_name='Data')
prices_df = pd.read_csv("DJIA Index.csv")
sentiment_df['date']= pd.to_datetime(sentiment_df['date'])
prices_df['Date']= pd.to_datetime(prices_df['Date'])
model_features = pd.merge(prices_df,sentiment_df,left_on = 'Date',right_on = 'date')

date = prices_df['Date']
Close = prices_df['Close']

Change = [0]* len(date)
Momentum = [0]* len(date)

for i in range (1,len(date)):
    if Close[i] > Close[i-1] :
        Momentum[i] = 1
        Change[i] = (Close[i]-Close[i-1])/Close[i-1]
    else :
        Momentum[i] = -1
        Change[i] = (Close[i-1] - Close[i]) / Close[i - 1]

#model_features  = pd.DataFrame({'Date':date, 'Close':Close,'Change':Change,'Momentum':Momentum})
#model_features = pd.merge(model_features,sentiment_df,left_on = 'Date',right_on = 'date')
model_features  = model_features.drop(columns = {'date','Close','Adj Close'})

expected = Momentum[1:]

def initialize_weights(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random.uniform(0,1) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random.uniform(0,1) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def activate(weights, inputs):
    #print(weights)
    activation = weights[-1]
    #print(weights[-1])
    for i in range(len(weights) - 1):
        #print("input:",inputs[i])
        #print("weight:",weights[i])
        activation += (weights[i] * inputs[i])
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

def sigmoid(activation):
    output = (np.exp(activation)-np.exp(-activation))/(np.exp(activation)+np.exp(-activation))
    return output

def sigmoid_derivation(output):
    der = output**2
    return(1 - der)

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            #print(inputs)
            activation = activate(neuron['weights'], inputs)
            #print("output:",activation)
            neuron['output'] = sigmoid(activation)
            #print("output after logistic:",neuron['output'])
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
            print
            for j in range(len(layer)):
                neuron = layer[j]
                #print(neuron['output'])
                #print(expected)
                errors.append(expected - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivation(neuron['output'])
            #print("delta:",neuron['delta'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] = neuron['weights'][j] - l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1]= neuron['weights'][-1] - l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, expected, l_rate):

    for i in range(len(train)):
        #print(train[i])
        output = forward_propagate(network, train[i])
        if(output[0]>0):
            final_output = 1
        else:
            final_output = - 1
        error = (expected[i] - output[0]) ** 2
        backward_propagate_error(network, expected[i])
        update_weights(network, train[i], l_rate)
       # print(network)
# Make a prediction with a network
def predict(network, row):
    output = forward_propagate(network, row)
    return output

n_layer = 3
input_nodes = model_features.shape[1]-1
hidden_nodes  = 3
output_nodes = 1
learning_rate = 0.5
network = initialize_weights(input_nodes,hidden_nodes,output_nodes)

fold = 1000
accuracy = []
increment = 300
while(fold<=1900):
    print(fold)

    train_df = model_features.iloc[:fold]
    train_df = train_df.drop(columns = {"Date"})
    train_df = np.asarray(train_df, dtype='float64')
    test_df = model_features.iloc[fold:].reset_index(drop = True)
    expected_test = expected[fold:]


    test_df = test_df.drop(columns = {"Date"})
    test_df = np.asarray(test_df, dtype='float64')
    expected_df = expected[:fold]

    train_network(network, train_df,expected_df,learning_rate)
    misclassification = 0
    total = 0
    for i in range(len(test_df)-1):
        total = total+1
        #print(test_df[i])
        #print(expected_test[i])
        prediction = predict(network, test_df[i])
        if(prediction[0]>0):
            final_prediction = 1
        else:
            final_prediction = -1
        if (expected_test[i] != final_prediction):
            misclassification = misclassification + 1
        Accuracy =(total - misclassification)*100/total

    accuracy.append(Accuracy)
    #print(accuracy)
    #print(total)
    fold = fold+increment

plot1 = pd.DataFrame(accuracy)
plot1['index'] = [i*100 for i in [10,13,16,19]]
fig, ax = plt.subplots()
plt.plot(plot1['index'],plot1[0])
ax.set_title('Accuracy using Back propogation')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Size of Training set');
plt.savefig('Variation of accuracy.png')
