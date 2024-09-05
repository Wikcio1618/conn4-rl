import numpy as np
from agents.agent import Agent
from board import Board


class RandomAgent(Agent):
    def choose_action(self, **kwargs):
        state = kwargs.get('state')
        valid_moves = []
        for i in range(Board.width):
            if state[0][i] == 0:
                valid_moves.append(i)
        return np.random.choice(valid_moves)
    
    def store_memory(self, experience) -> None:
        pass