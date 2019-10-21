from collections import namedtuple
from sc2rl.memory.TrajectoryBase import TrajectoryBase

experience_spec = namedtuple('exp_args',
                             ["state", "action", "reward", "next_state", "done"],
                             defaults=tuple([list() for _ in range(5)]))


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
                self._trajectory.append(self.spec(state, action, reward, next_state, done))
