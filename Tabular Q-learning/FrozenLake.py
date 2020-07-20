import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
gamma = 0.9
alpha = 0.2
test_episodes = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.values = collections.defaultdict(float)
        self.state = self.env.reset()
    
    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, __ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state)

    def best_value_and_action(self,state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values.get[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a,r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + gamma*best_v
        old_val = self.values[(s,a)]
        self.values[(s,a)] = old_val*(1-alpha) + new_val*alpha

    def play_episode(slef,env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done: break
            state = new_state
        return total_reward

    