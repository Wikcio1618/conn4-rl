from typing import Any, List
from numpy.typing import ArrayLike

from agents.agent import Agent
from board import Board


class Game:
    @classmethod
    def play_exp_game(cls, agents:List[Agent]):
        """ 
        Plays a game and adds the experience agents' `replay_buffer`s\n
        The first player in `agents` starts
        """

        if len(agents) != 2:
            raise ValueError(f"There must be exactly 2 playars in the game. Provided {len(agents)}")
        
        board = Board()
        piece_tag = 2 # tag is either 1 or 2. It is a board piece tag and is used to index agents list
        done = False
        states, actions, rewards, dones = [], [], [], []

        while not done:
            piece_tag = 1 if piece_tag == 2 else 2

            state = Agent.get_board_representation(board, piece_tag)
            action = agents[piece_tag-1].choose_action(state=state, piece_tag=piece_tag, policy='greedy')
            reward = board.drop_piece(action, piece_tag)
            
            # update previous player if he lost or drew
            if reward == Board.rewards_dict['win']:
                rewards[-1] = Board.rewards_dict['loss']
                dones[-1] = True
            elif reward == Board.rewards_dict['draw']:
                rewards[-1] = Board.rewards_dict['draw']
                dones[-1] = True
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(reward != 0)
            done = dones[-1]
        
        # minus is there because its the other guys' perspective
        states.append(-state)
        states.append(state)

        for i in range(len(rewards)):
            agents[i % 2].store_memory((states[i], actions[i], rewards[i], states[i+2], dones[i]))

    # def play_eval_game(main_agent:Agent, num_games = 100) -> int:
    #     """
    #     # Returns:
    #     total reward gained divided by `num_games`. It's from range [`-1`, `1`]
    #     """
        
    #     board = Board()
    #     test_agent = Agent(main_model = ConnectFourNN(), piece_tag = 1 if main_agent.piece_tag == 2 else 2, device = device, board = board)
    #     main_agent.board = board
    #     curr_agent = test_agent

    #     total_reward = 0
    #     for _ in range(num_games):
    #         done = False
    #         board.reset_board()
            
    #         while not done:
    #             curr_agent = main_agent if curr_agent != main_agent else test_agent
    #             eps = 0 if curr_agent == main_agent else 1

    #             state = curr_agent.get_board_state()
    #             action = curr_agent.choose_action(state, type='greedy', eps = eps)
    #             reward = curr_agent.perform_action(action)

    #             done = (reward != 0)
            
    #         if reward == Board.rewards_dict['win']:
    #             total_reward += 1 if curr_agent == main_agent else -1
        
    #     return total_reward / num_games


        
        
        
        
        