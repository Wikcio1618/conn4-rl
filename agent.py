import random
import numpy as np
from board import Board
from replay_buffer import ReplayBuffer
import torch.nn as nn
import torch
import torch.nn.functional as F

class Agent:
    def __init__(self, main_model:nn.Module, piece_tag:int, board:Board=None):
        self.main_model = main_model
        if board is not None:
            self.board = board
        self.piece_tag = piece_tag

    def drop_piece(self, col):
        self.board.drop_piece(col, self.piece_tag)
    

    def get_board_state(self):
        perspective_board = np.zeros_like(self.board.pieces)
        perspective_board[self.board.pieces == self.piece_tag] = 1
        perspective_board[(self.board.pieces != self.piece_tag) * (self.board.pieces != 0)] = -1
        return perspective_board
    
    def choose_action(self, state, type = 'greedy', eps = None):
        """
        # Returns:
        `action` chosen according to `type` policy based on `state`\n
        """

        if type not in ('greedy', 'softmax'):
            raise ValueError("Choose one of types: 'greedy', 'softmax'")
        
        if type == 'greedy' and eps is None and eps >= 0 and eps <= 1:
            raise ValueError(f"For greedy action choice type you have to set `eps` from range [0, 1]. Now 'eps'={eps}")
        
        if type == 'greedy':
            if np.random.rand() < eps:
                valid_moves = []
                for i in range(state.shape[1]):
                    if state[0][i] == 0:
                        valid_moves.append(i)
                return np.random.choice(valid_moves)
            else:
                with torch.no_grad():
                    actions = self.get_actions_pred(state)
                    
                    best_action = actions.argmax().item()
                    return best_action
                
        elif type == 'softmax':
            # needs to be tested
            preds_proba = self.get_actions_pred(state)
            thresh = 0
            for action, proba in enumerate(preds_proba):
                thresh += proba
                if random.random() < thresh:
                    return action
            


    def perform_action(self, action:int):
        """
        # Returns:
        `reward`
        """

        reward = self.board.drop_piece(col = action, piece_tag = self.piece_tag)
        return reward

    def get_actions_pred(self, state:np.ndarray):
        self.main_model.eval()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            preds = self.main_model(state_tensor)
            # https://ai.stackexchange.com/questions/2980/how-should-i-handle-invalid-actions-when-using-reinforce
            masked_preds = torch.log(torch.tensor(self.board.get_valid_moves_mask())) + preds
            preds_proba = F.softmax(masked_preds.squeeze(), dim=0)
        return preds_proba