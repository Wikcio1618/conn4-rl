from collections import deque, namedtuple
import random

# Experience = namedtuple("Experience", ['state', 'action', 'reward', 'new_state'])

class ReplayBuffer:
    def __init__(self, maxlen:int = 10**4) -> None:
        self.buffer = deque(maxlen=maxlen)

    def add(self, experience) -> None:
        deque.append(experience)

    def extend(self, experiences:list) -> None:
        self.buffer.extend(experiences)

    def clear_oldest(self, N:int):
        for _ in range(N):
            self.buffer.popleft()

    def sample(self, batch_size:int = 32):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)