from collections import deque
from copy import deepcopy


class TrajectoryBase:

    def __init__(self, spec, gamma, max_len=100000):
        self.spec = spec
        self.gamma = gamma
        self._trajectory = deque(maxlen=max_len)
        self.returns = None

    @property
    def length(self):
        return len(self._trajectory)

    def push(self, sample):
        assert self.spec._fields == sample._fields
        self._trajectory.append(sample)

    def __getitem__(self, index):
        return self._trajectory[index]

    def reset(self):
        self._trajectory.clear()

    def get_n_samples_from_i(self, n, i):
        ret_dict = deepcopy((self.spec)._asdict())

        for j in range(i, i + n):
            sample = sample._asdict()
            for field, val in sample.items():
                ret_dict[field].append(val)

        return ret_dict

    def compute_return(self):
        rewards = self.rewards
        returns = np.zeros_like(rewards, dtype=float)
        # set the last return as the last reward
        returns[-1] = rewards[-1]

        # Iterating over rewards to compute returns in backward
        for i, reward in enumerate(reversed(rewards[:-1])):
            backward_index = self.len_trajectory - 1 - i
            returns[backward_index - 1] = rewards[backward_index - 1] + self.gamma * returns[backward_index]

        self.returns = returns
        self.discounted = True

