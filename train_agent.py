
## Script used to train the agent 

import pickle
import gzip
import numpy as np

from agent import Agent

def read_data():
    with gzip.open('./data/X.npy.gzip', 'rb') as g:
        X = np.load(g)
    with gzip.open('./data/y.npy.gzip', 'rb') as g:
        y = np.load(g)
    return X, y

def split(X, y, train_size = 0.8):
    idx = int(train_size * X.shape[0]) - 1
    X_train = X[0:idx]
    y_train = y[0:idx]
    X_valid = X[idx:]
    y_valid = y[idx:]
    return X_train, y_train, X_valid, y_valid
    
## Loading and splitting the data in training and validation set

X, y = read_data()
X_train, y_train, X_valid, y_valid = split(X, y)

## Generate the agent 
agent = Agent()

## Train the model

history = agent.train(X_train, y_train, X_valid, y_valid, 10, 64)

## Saving the model weights and the training history

agent.save_model('./model/weigths.h5')

with open("./model/history.pkl", "wb") as f:
    pickle.dump(history.history, f)