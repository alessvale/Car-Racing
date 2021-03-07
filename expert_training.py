
# Script used to generate the trajectory data

import gym
import random
import numpy as np

env_name = "CarRacing-v0"
env = gym.make(env_name)

####### Functions to store data

import pickle
import os


def store_data(data, episode, datasets_dir="./expert_data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'expert_data' + str(episode) + '.pkl')
    f = open(data_file,'wb')
    pickle.dump(data, f)


#####


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

 
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
   
    isopen = True
    
    ## Episode loop##
    EP_MAX = 15
    episode = 0 
    
    while isopen:
        
        episode += 1
        
        state = env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        
        ## Create a sample dictionary for each episode
        
        samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
        }
        
       
        
        ### State loop ####
        
        while True:
            next_state, r, done, info = env.step(a)
            total_reward += r
            
            samples["state"].append(state)            
            samples["action"].append(np.array(a[:3]))    
            samples["next_state"].append(next_state)
            samples["reward"].append(r)
            samples["terminal"].append(done)
            
            state = next_state
            steps += 1
            
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            
            
            isopen = env.render()
            if done or restart or isopen == False:
                break
        if not restart:   
            
            
            print('... saving data')
            store_data(samples, episode, "./expert_data")
            
        
        if (episode > EP_MAX):
            env.close()
            break
                
    env.close()