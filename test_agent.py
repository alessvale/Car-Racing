
## Script used to test the agent performance


import gym
import random
import numpy as np

from agent import Agent

env_name = "CarRacing-v0"
env = gym.make(env_name)


## Store results for stats

import pickle
import os
import json


def save_results(episode_rewards, episode, results_dir="./results_test"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

     # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_steps"] = len(episode_rewards)
    results["total_reward"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = os.path.join(results_dir, "results_test" + str(episode) + ".json")
    fh = open(fname, "w")
    json.dump(results, fh)
    print("Saved!")

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
    episode = 0 
    
    ## Load agent
    
    agent = Agent()
    agent.load_weights("./model/weights.h5")
    
    while isopen:
        
        episode += 1
        
        state = env.reset()
        agent.begin_new_episode()
        
        total_reward = 0.0
        steps = 0
        restart = False
        
        ## Store rewards  
        rewards = []
        
        ### State loop ####
        
        while True:
            
            agent_action = agent.get_action(state)
            
            next_state, r, done, info = env.step(agent_action)
            total_reward += r
            
            state = next_state
            steps += 1
            
            rewards.append(total_reward)
            
            if steps % 1000 == 0 or done:
                
                print("\naction " + str(["{:+0.2f}".format(x) for x in agent_action]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            
            
            isopen = env.render()
            if done or restart or isopen == False:
                #save_results(rewards, episode,  "./results_test")
                break
                
        
    env.close()