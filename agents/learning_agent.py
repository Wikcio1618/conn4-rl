import numpy as np
import torch
from replay_buffer import ReplayBuffer
import torch.nn as nn
import torch.nn.functional as F
from agents.nn_agent import NNAgent

class LearningAgent(NNAgent):
    loss_class = nn.HuberLoss()

    def __init__(self, main_model:nn.Module, target_model:nn.Module, replay_buffer:ReplayBuffer, optimizer:torch.optim.Optimizer, eps=0, device='cpu'):
        super().__init__(main_model, eps, device)
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer

    def store_memory(self, experience) -> None:
        if self.replay_buffer is not None:
            self.replay_buffer.add(experience)

    def train_step(self, batch_size, gamma):
        """Returns: loss"""
        if len(self.replay_buffer) < batch_size:
            return 0
        
        self.optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), max_norm=1.0)

        self.main_model.train()

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # equal_flags = [np.array_equal(states[i], next_states[i]) for i in range(len(states))]

        states = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.int8).to(self.device)

        q_values = self.main_model(states).gather(1, actions).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1-dones)

        loss = self.loss_class(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()
    
    def update_target_model(self) -> None:
        self.target_model.load_state_dict(self.main_model.state_dict())
