import gym
import random
class RandomActionWrapper(gym.ActionWrapper):
    #initializing the class
    def __init__(self,env,epsilon=0.2):
        super(RandomActionWrapper,self).__init__(env)
        self.epsilon = epsilon

    #now we overide the the parent class and modify the action sent by 
    #agent to perform to the environment

    def action(self,action):
        if random.random()<self.epsilon:
            print("Random action!")
            return self.env.action_space.sample()
        return action

if __name__=="__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0")) #creating the environment and passing it to the new class
    
    obs = env.reset()
    total_reward =0.0
    total_steps = 0

    while 1:
        obs, reward, done, _ =env.step(0) # always passing 0 as action
        total_reward +=reward
        total_steps += 1
        if done: break

    print("Episode done in %d steps, total reward %2.f" %(total_steps,total_reward))

