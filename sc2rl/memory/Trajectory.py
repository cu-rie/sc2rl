from collections import namedtuple
import numpy as np

from sc2rl.memory.TrajectoryBase import TrajectoryBase

experience_spec = namedtuple('exp_args',
                             ["state", "action", "reward", "next_state", "done", "ret"],
                             defaults=tuple([list() for _ in range(6)]))


class Trajectory(TrajectoryBase):

    def __init__(self,
                 gamma: float,
                 max_len: int = 1000):
        spec = experience_spec
        super(Trajectory, self).__init__(spec=spec,
                                         gamma=gamma,
                                         max_len=max_len)

    def push(self, sample):
        assert self.spec._fields == sample._fields
        self._trajectory.append(sample)
        done = sample.done
        if done:
            # check whether the graph actually has nodes
            # when the last frame of each episode has 0 units
            # delete the frame and set the frame before last set as new last frame
            if sample.next_state.number_of_nodes() == 0:
                self._trajectory.pop()
                sample = self._trajectory.pop()
                state = sample.state
                action = sample.action
                reward = sample.reward
                next_state = sample.next_state
                done = True
                ret = sample.ret
                self._trajectory.append(self.spec(state, action, reward, next_state, done, ret))
            self.compute_return()

    def compute_return(self):
        rewards = [sample.reward for sample in self._trajectory]
        returns = np.zeros_like(rewards, dtype=float)

        # set the last return as the last reward
        returns[-1] = rewards[-1]

        # Iterating over rewards to compute returns in backward
        for i, reward in enumerate(reversed(rewards[:-1])):
            backward_index = self.length - 1 - i
            returns[backward_index - 1] = rewards[backward_index - 1] + self.gamma * returns[backward_index]

        # Set return values to the samples
        for i in range(self.length):
            sample = self._trajectory.popleft()

            state = sample.state
            action = sample.action
            reward = sample.reward
            next_state = sample.next_state
            done = sample.done
            ret = returns[i]
            self._trajectory.append(self.spec(state, action, reward, next_state, done, ret))

        self.discounted = True
