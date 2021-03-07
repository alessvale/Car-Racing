import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, Input

from utils import preprocessing, id_to_action

from collections import deque

### Agent class

MIN_FRAMES = 50

class Agent:
    
    def __init__(self):
        
        self.accelerate = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.model = self.make_model()
        
    
    def make_model(self):
        
        ## Width and height of the pre-processed images
        w = 80
        h = 90
        
        ## Constructing the convolutional neural network
        
        model = Sequential()
        model.add(Input(shape = (w, h, 1)))
        model.add(Conv2D(kernel_size=(5,5), filters=16, strides=(4, 4), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(kernel_size=(3,3), filters=32, strides=(2, 2), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dense(5, activation='softmax')) 
        
        return model
    
   
    def load_weights(self, filename):
        
        self.model.load_weights(filename)
        
    def save_model(self, filename):
        
        self.model.save_weights(filename)
    
    def train(self, X, y, X_valid, y_valid, epochs, batch_size):
        
        optimizer = tf.keras.optimizers.Adam(5 * 0.0001)
        self.model.compile(optimizer=optimizer, loss= "categorical_crossentropy", metrics = [tf.metrics.CategoricalAccuracy()])

        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, steps_per_epoch= X.shape[0] // batch_size, validation_data = (X_valid, y_valid), shuffle = True)
        
        return history
    
    ## Keeps track of the last 100 actions
    
    def begin_new_episode(self):
          
        self.action_history = deque(maxlen=100)
        self.overwrite_actions = []
        self.action_counter = 0
    
    
    ## Check if all the last actions are the same and they are not accelerate
    
    def check_idle(self):
        
        ## Check if all actions are the same
        
        fa = self.action_history[0]
        for a in self.action_history:
            if not np.all(a==fa):
                return False
            if np.all(a == self.accelerate):
                return False
        
        # Since the car is idleing, we release the break, and fill the array of actions to overwrite
        
        fa[2] = 0.0  # release break in case it is on
        overwrite_cycles = 2
        one_cicle = 5 * [fa] + 5 * [self.accelerate]
        self.overwrite_actions = overwrite_cycles * one_cicle
        return True
        
        
        
    def get_action(self, state):
        
        # Always accelerate for the first MIN_FRAMES frames
        
        if self.action_counter < MIN_FRAMES:
            self.action_history.append(self.accelerate)
            self.action_counter += 1
            return self.accelerate
        
        # In the case the agent is stuck for too long, we overwrite the network
        
        if len(self.overwrite_actions) > 0:
            
            action = self.overwrite_actions.pop()
            self.action_history.append(action)
            return action
        
        
        if self.check_idle():
            print("Agent is idleing")
        
        state_reshaped = np.expand_dims(preprocessing(state), axis = 2)
        state_reshaped = np.array([state_reshaped])
        y = self.model.predict(state_reshaped)
        w = np.array([0.3, 1.0, 1.0, 1.0, 1.0]) # Weights used to recalibrate the network predictions 
        y *= w
        a_id = np.argmax(y)
        action = id_to_action(a_id)
        self.action_history.append(action)
        return action
  