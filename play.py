import tkinter as tk
from tkinter import messagebox
from functools import partial
import numpy as np
from board import Board
from model import ConnectFourNN
import torch
import sys

class ConnectFourGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Connect Four")

        self.board = Board()
        self.model = ConnectFourNN()
        # self.model.set_parameters_from_file('10.01.24/check3.txt')
        self.model = torch.load(sys.argv[1])

        self.buttons = []
        self.fields = []
        for col in range(self.board.width):
            button = tk.Button(self.root, text=str(col + 1), command=partial(self.make_move, col))
            button.grid(row=0, column=col)
            self.buttons.append(button)

        for row in range(self.board.height):
            for col in range(self.board.width):
                button = tk.Button(self.root, text="", width=4, height=2)
                button.grid(row=row+1, column=col)
                self.fields.append(button)

    def make_move(self, col):
        if self.board.is_valid_move(col) and not self.board.is_board_full():
            reward = self.board.drop_piece(col, 1)  # Assuming player 1 for the user
            self.update_gui()

            # Check for a winner
            if reward == Board.rewards_dict['win']:
                messagebox.showinfo("Game Over", "You win!")
                self.reset_game()

            # Check for a tie
            elif reward == Board.rewards_dict['draw']:
                messagebox.showinfo("Game Over", "It's a tie!")
                self.reset_game()

            # AI's move (replace with your AI logic)
            else:
                # Example: AI makes a random valid move
                board_tensor = torch.tensor(self.board.pieces, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                # Put the model in evaluation mode
                self.model.eval()
                # Make a forward pass to obtain the predictions
                with torch.no_grad():
                    predictions = self.model(board_tensor)
                ai_col = np.argmax(predictions.numpy())

                if self.board.is_valid_move(ai_col):
                    reward = self.board.drop_piece(ai_col, 2)  # Assuming player 2 for the AI
                    self.update_gui()
                    # Check for AI win
                    if reward == Board.rewards_dict['win']:
                        messagebox.showinfo("Game Over", "AI wins!")
                        self.reset_game()
                else:
                    messagebox.showinfo("Game Over", "AI played non-valid move!")
                    self.reset_game()

        else:  
            messagebox.showinfo("Think Again", "Invalid move!!")
                

    def update_gui(self):
        for col in range(self.board.width):
            for row in range(self.board.height):
                field_idx = row * self.board.width + col
                if self.board.pieces[row, col] == 1:
                    self.fields[field_idx].config(bg='red')
                elif self.board.pieces[row, col] == 2:
                    self.fields[field_idx].config(bg='yellow')
                elif self.board.pieces[row, col] == 0:
                    self.fields[field_idx].config(bg='white')

    def reset_game(self):
        self.board = Board()
        self.update_gui()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = ConnectFourGUI()
    gui.run()
