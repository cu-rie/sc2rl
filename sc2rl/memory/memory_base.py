from collections import deque
import numpy as np
from sc2rl.memory.trajectory import Trajectory


class MemoryBase:
    """
        An abstract class for memory
    """

    def push(self, *args, **kwargs):
        """
            abstract function that is for appending experience tuple to the memory
        """
        pass

    def sample(self, *args, **kwargs):
        """
            abstract function that is for getting samples from the memory
        """

    def reset(self, *args, **kwargs):
        """
            abstract function that is for resetting memory
        """
        pass


class EpisodicMemory(MemoryBase):

    def __init__(self, max_n_episodes, spec, gamma, max_traj_len=100000):
        self.max_n_episodes = max_n_episodes
        self.trajectories = deque(maxlen=max_n_episodes)

        self.sepc = spec
        self.gamma = gamma
        self.max_traj_len = max_traj_len
        self._cur_traj = Trajectory(spec=spec, gamma=self.gamma, max_len=max_traj_len)

    def len_trajectories(self):
        return np.array([traj.length for traj in self.trajectories])

    def reset(self):
        self.trajectories.clear()