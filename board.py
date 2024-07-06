import numpy as np

class Board:
    def __init__(self, height=6, width=7, draw_reward = 0.3, win_reward = 1):
        self.height = height
        self.width = width
        self.draw_reward = draw_reward
        self.win_reward = win_reward
        self.pieces = np.zeros((height, width))

    def check_winner(self, move:tuple, player) -> bool:
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
            if self.check_line(line, player):
                return True
            
        return False

    def check_line(self, line, player):
    # Check if there are four consecutive pieces of the same player in the line
        count = 0
        for cell in line:
            if cell == player:
                count += 1
                if count == 4:
                    return True
            else:
                count = 0
        return False
    
    def is_valid_move(self, col):
        if self.pieces[0, col] == 0:
            return True
        
        return False
    
    def is_board_full(self):
        return np.all(self.pieces[0] != 0)
    
    def drop_piece(self, col, player:int):
        """
        returns:
        next_state, reward, done
        """
        assert col >= 0 and col < self.width

        for i in range(self.height - 1, -1, -1):
            if self.pieces[i, col] == 0:
                self.pieces[i, col] = player
                row = i
                break
        
        if self.check_winner((row, col), player):
            reward = self.win_reward
        elif self.is_board_full():
            reward = self.draw_reward
        else: 
            reward = 0

        return (self.pieces.copy(), reward, (reward != 0))
    
    def reset_board(self) -> None:
        self.pieces = np.zeros((self.height, self.width))