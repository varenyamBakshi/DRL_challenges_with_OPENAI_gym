import gym
import collections
from tensorboardX import SummaryWriter

ENV_name = "FrozenLake-v0"
gamma = 0.9
test_episodes = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_name)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float) #it will assign default value 0.0 to a new key
        self.transits = collections.defaultdict(collections.Counter) # to keep a count of hashable items
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for i in range(count):
            action = self.env.action_space.sample() #choosing a random action
            #updating the reward and transition tables
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state,action,new_state)] = reward
            self.transits[(self.state,action)][new_state]+=1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        target_counts = self.transit[(state,action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.reward[(state,action,tgt_state)]
            action_value += (count/total)*(reward + gamma*self.values[tgt_state])
        return action_value

    def select_action(self,state):  #choose the best action to take for given state
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value((state, action))
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = slef.select_action(state)
            new_state, reward, is_done, __ = env.step(action)
            self.rewards[(state,action,new_state)] = reward
            self.transit[(state,action)][new_state] += 1
            total_reward += reward
            if is_done: break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_value = [self.calc_action_value(state,action) for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)

    