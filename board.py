import numpy as np

class Board:

    rewards_dict = {
        'win': 1,
        'draw': 0.3,
        'loss': -1,
        'illegal_move': -1,
        'valid_move': 0
    }

    def __init__(self, height=6, width=7):
        self.height = height
        self.width = width
        self.pieces = np.zeros((height, width), dtype=np.int8)

    def check_winner(self, move:tuple, piece_tag) -> bool:
        """
        move - (row, col)
        True - winning move
        False - not 
        """
        row, col = move[0], move[1]

        lines = []
        lines.append(self.pieces[row, :])
        lines.append(self.pieces[:, col])
        lines.append(np.diagonal(self.pieces, offset=(col-row)))
        lines.append(np.diagonal(np.fliplr(self.pieces), offset=(self.width - col - row - 1)))
        
        for line in lines:
            if self.check_line(line, piece_tag):
                return True
            
        return False

    def check_line(self, line, piece_tag):
    # Check if there are four consecutive pieces of the same player in the line
        count = 0
        for cell in line:
            if cell == piece_tag:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
        return False
    
    def is_valid_move(self, col):
        return self.pieces[0, col] == 0
    
    def is_board_full(self):
        return np.all(self.pieces[0] != 0)
    
    def get_valid_moves_mask(self):
        return np.array([self.is_valid_move(col) for col in range(self.width)], dtype=np.int8)
    
    def drop_piece(self, col, piece_tag) -> tuple:
        """
        returns:
        `reward`: val - according to `reward_dict`\n
        """
        assert col >= 0 and col < self.width

        if not self.is_valid_move(col):
            return Board.rewards_dict['illegal_move']

        for i in range(self.height - 1, -1, -1):
            if self.pieces[i, col] == 0:
                self.pieces[i, col] = piece_tag
                row = i
                break
        
        if self.check_winner((row, col), piece_tag):
            reward = Board.rewards_dict['win']
        elif self.is_board_full():
            reward = Board.rewards_dict['draw']
        else:
            reward = Board.rewards_dict['valid_move']

        return reward
    
    def reset_board(self) -> None:
        self.pieces = np.zeros_like(self.pieces)