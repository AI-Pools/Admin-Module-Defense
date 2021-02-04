import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from collections import namedtuple
import numpy as np
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
BATCH_SIZE = 128     
GAMMA = 0.999        
LEARNING_RATE = 0.001
NB_EPISODE = 50_000
NB_ITERATION = 500

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class Memory(object):

    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: list = []
        self.position: int = 0 
    
    def clear(self):
        self.memory: list = []
        self.position: int = 0

    def push(self, *args: list):
        """push a transition"""
        if len(self.memory) < self.capacity:               
            self.memory.append(None)                        
        self.memory[self.position] = Transition(*args)      
        self.position = (self.position + 1) % self.capacity 

    def batch(self, batch_size):
        return self.memory
    
    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, env, neural, optimizer):
        self.memory = Memory(2000)
        self.env = env
        self.neural = neural
        self.optimizer = optimizer
        self.init()

    def init(self):
        pass
    
    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def take_action(self, state) -> [int]:
        logits_v = self.neural(torch.Tensor(state))
        prob_v = F.softmax(logits_v, dim=0)
        action = prob_v.multinomial(num_samples=1)
        return action.item()

    def calc_qvals(self, rewards):
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= GAMMA
            sum_r += r
            res.append(sum_r)
        return list(reversed(res))

    def learn(self):
        transitions = self.memory.batch(BATCH_SIZE)
        # On passe d'un tableau de transition a une transition de tableau
        # On met * pour passer l'arg à la fonction zip puis * pour unzip sous forme de list
        batch: Transition = Transition(*zip(*transitions))
        batch_qvals = []
        batch_qvals.extend(self.calc_qvals(batch.reward))

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch.state)
        batch_actions_t = torch.LongTensor(batch.action)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        logits_v = self.neural(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(states_v)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()
        self.memory.clear()
       

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    neural = PGN(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(neural.parameters(), lr=LEARNING_RATE)
    agent: Agent = Agent(env, neural, optimizer)
    episode_rewards = []
    for episode in range(NB_EPISODE):
        new_state = env.reset()
        done = False
        i = 0
        for i in range(NB_ITERATION):
            env.render()
            state = new_state
            action = agent.take_action(state)
            new_state, reward, done, _ = env.step(action)
            if done == True:
                if i >= 199:
                    reward = 2
                else :
                    reward = -5
                agent.memory.push(state, action, None, reward)
                episode_rewards.append(reward)
                print(f"episode: {episode}, iteration {i}, mean {np.mean(episode_rewards)}")
                agent.learn()
                break
            agent.memory.push(state, action, new_state, reward)
            episode_rewards.append(reward)
        print(f"episode: {episode}, iteration {i}, mean {np.mean(episode_rewards)}") 