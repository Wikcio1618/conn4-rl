import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent import Agent
from board import Board
from model import ConnectFourNN


class NNAgent(Agent):
    def __init__(self, main_model:nn.Module, eps=0, device='cpu'):
        self.main_model = main_model
        self.eps = eps
        self.device = device
    
    def choose_action(self, **kwargs):
        """
        `state`: ndarray|Tensor\n
        `policy`\n
        `eps`:float\n
        # Returns:
        `action` chosen according to `policy` policy based on `state`\n
        """

        state:np.ndarray|torch.Tensor = kwargs.get('state')
        policy:str|None = kwargs.get('policy')
        eps:int|None = kwargs.get('eps')
        if policy is None:
            policy = 'greedy'

        if policy not in ('greedy', 'softmax'):
            raise ValueError("Choose one of policies: 'greedy', 'softmax'")
        
        if eps is None:
            eps = self.eps
        
        if policy == 'greedy' and (eps < 0 or eps > 1):
            raise ValueError(f"For greedy action choice policy you have to set `eps` from range [0, 1]. Now 'eps'={eps}")
        
        if policy == 'greedy' or eps == 1:
            if np.random.rand() < eps:
                valid_moves = []
                for i in range(Board.width):
                    if state[0][i] == 0:
                        valid_moves.append(i)
                return np.random.choice(valid_moves)
            else:
                with torch.no_grad():
                    actions = self.get_masked_actions_proba(state)
                    best_action = actions.argmax().item()
                    return best_action
                
        elif policy == 'softmax':
            # needs to be tested
            preds_proba = self.get_masked_actions_proba(state)
            thresh = 0
            for action, proba in enumerate(preds_proba):
                thresh += proba
                if random.random() < thresh:
                    return action
    

    def get_masked_actions_proba(self, state:np.ndarray):
        """ 
         Takes the `state`, passes it through `self.model` to get `q-values vector`\n
         uses smart masking by adding to it logarithm of `mask of valid moves`\n
         finally softmax is applied so it returns probabilities
        """
        self.main_model.eval()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.main_model(state_tensor)
            # https://ai.stackexchange.com/questions/2980/how-should-i-handle-invalid-actions-when-using-reinforce
            masked_preds = torch.log(torch.tensor(Board.get_valid_moves_mask(state)).to(self.device)) + preds
            preds_proba = F.softmax(masked_preds.squeeze(), dim=0)
        return preds_proba
    
    def store_memory(self, experience) -> None:
        pass

    def get_action_qval(self, states:np.ndarray|None = None, actions = (2, 2, 2)):
        """
        For a list of `states`, return best q_values estimated by the `model` 
        """

        if states is None:
            states = np.array([
                np.array([
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ], dtype=np.int8),
                np.array([
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, -1, 0, 0, 0],
                    [0, 0, 1, -1, -1, 0, 0],
                ], dtype=np.int8),
                np.array([
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                    [1, 0, -1, -1, -1, 0, 0],
                ], dtype=np.int8)
                ])

        states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(1).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)

        with torch.no_grad():
            q_values = self.main_model(states_tensor).gather(1, actions).cpu().numpy()
        return q_values

    @classmethod
    def from_file(cls, path:str, eps=0, device='cpu'):
        model = ConnectFourNN().to(device)
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        return cls(model, eps, device)