import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")   #creating the environment
    #initializing the parameters
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
    #start interacting with the environment
    while True:
        action = env.action_space.sample() #accept a random action
        obs, reward, done, _ = env.step(action) #performing an action an recieving rewards, obervation
                                                # and also noting whether the episode is over or not
        total_reward+=reward
        total_steps+=1
        if done: break

    print("Episode done in %d steps, total reward %2.f" %(total_steps,total_reward))



