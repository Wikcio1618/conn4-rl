import tkinter as tk
from tkinter import messagebox
from functools import partial
import numpy as np
from agent import Agent
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
        self.model = torch.load(sys.argv[1], map_location=torch.device('cpu'))
        self.ai_agent:Agent = Agent(self.model, piece_tag=2, board=self.board, device='cpu')

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

        self.q_labels = []
        for col in range(self.board.width):
            q_label = tk.Label(self.root, text="")
            q_label.grid(row = 2 + self.board.height, column=col)
            self.q_labels.append(q_label)

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
                board_state = self.ai_agent.get_board_state()
                action = self.ai_agent.choose_action(board_state, eps=0)
                actions_pred = self.ai_agent.get_actions_pred(board_state)
                for col in range(self.board.width):
                    self.q_labels[col].config(text = f"{actions_pred.squeeze()[col].item():.2f}")


                if self.board.is_valid_move(action):
                    reward = self.ai_agent.perform_action(action)
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
                    self.fields[field_idx].config(bg='yellow')
                elif self.board.pieces[row, col] == 2:
                    self.fields[field_idx].config(bg='red')
                elif self.board.pieces[row, col] == 0:
                    self.fields[field_idx].config(bg='white')

    def reset_game(self):
        self.board.reset_board()
        for col in range(self.board.width):
            self.q_labels[col].config(text = "")
        self.update_gui()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("You need to provide 1 argument: path to .pth model")
    else:
        gui = ConnectFourGUI()
        gui.run()
