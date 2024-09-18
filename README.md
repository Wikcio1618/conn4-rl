# Reinforcement Learning project - Connect4 with Deep Q-Network (DQN)
This project was my free-time, hobby affair. 

### Aims
* Practicing DQN algorithm
* Creating AI agent that can beat me in Connect4 game
* Practicing operational duties in model developement (fine-tuning etc.)

### Project structure
In _agents_ folder, there are classes for different types of models. Most important one is Learning Agent, who is able to play games and backpropagate gradients on its network
_play.py_ file is to be run to play a game with chosen agent. First command line argument should be path to .pth/.pt file
_lab.ipynb_ file is where training loop is implemented

### Obstacles and outcomes
* The greatest obstacle for me was lack of good GPU and long execution time
* Choice of learning process. For long time I made the agent play against itself and have one _replay buffer_. Then I switched to 3 independent agents playing against each other. It prevented overfitting and allowed for divergent experiences  
* _best_ folder contains my best models as of september 2024. Those aren't satisfactory, but are able to block 4-in-a-row (or in column or diagonal) threats. First couple of moves are decent strategic moves, but later they just naively attack and block.
* When opportunity comes I want to try: longer learning time, maybe greater lr than 4e-6, and eps parameter going to 1 again in the middle of learning
