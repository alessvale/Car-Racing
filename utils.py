import numpy as np

## Dictionary which maps actions as understood by the OpenAI environment to action the agent's model classifies

actions_dict = np.array([
    [ 0.0, 0.0, 0.0],  # go straight
    [ 0.0, 1.0, 0.0],  # accelerate
    [ 1.0, 0.0, 0.0],  # turn right
    [ 0.0, 0.0, 0.8],  # brake
    [-1.0, 0.0, 0.0],  # turn left
], dtype=np.float64)


## Pre-process images: more precisely, we recolor the grass, perform cropping, and invert the colors 

def preprocessing(X):
    ## Substituting the green patches with the color white
    
    mask = np.all(X == [102, 229, 102], axis = 2)
    X[mask] = [255, 255, 255]
    mask = np.all(X == [102, 204, 102], axis = 2)
    X[mask] = [255, 255, 255]
    
    # Scaling data
    X_bw = X/255.
    
    # Cropping and inverting colors 
    X_bw = 1.0 - X_bw[0:80, 0:90, 1]
    
    
    return X_bw

## Maps OpenAI actions to one-hot encoded vectors suitable for the neural network

def action_to_id(arr):
    one_hot = []
    idxs = []
    for a in arr:
        c = np.zeros(5)
        idx = np.where(np.all(actions_dict==a, axis=1))
        c[idx] = 1.0
        one_hot.append(c)
        idxs.append(int(idx[0][0]))
    return np.array(one_hot), idxs

## Maps id to actions understood by OpenAI environment

def id_to_action(idx):
    return actions_dict[idx]
 
## Check invalid actions, i.e. those not appearing in the action dictionary
 
def check_valid(action):
        idx = np.where(np.all(actions_dict==action, axis=1))
        if idx[0].size == 0:
            return False
        else:
            return True