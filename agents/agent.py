from abc import ABC, abstractmethod
import random
import numpy as np
import torch.types
from board import Board
from replay_buffer import ReplayBuffer
import torch.nn as nn
import torch
import torch.nn.functional as F

class Agent(ABC):
    @abstractmethod
    def choose_action(self, *args, **kwargs) -> int:
        pass

    @abstractmethod
    def store_memory(self, experience) -> None:
        pass

    @classmethod
    def get_board_representation(cls, board:Board, piece_tag:int):
        perspective_board = np.zeros_like(board.pieces)
        perspective_board[board.pieces == piece_tag] = 1
        perspective_board[(board.pieces != piece_tag) * (board.pieces != 0)] = -1
        return perspective_board