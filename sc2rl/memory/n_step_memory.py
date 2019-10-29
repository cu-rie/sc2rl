from collections import namedtuple

import numpy as np

from sc2rl.memory.memory_base import EpisodicMemory
from sc2rl.memory.Trajectory import Trajectory
from sc2rl.config.ConfigBase_refac import ConfigBase


class NstepInputMemoryConfig(ConfigBase):
    def __init__(self, memory_conf=None):
        super(NstepInputMemoryConfig, self).__init__(memory_conf=memory_conf)

        spec = namedtuple('exp_args', ["state", "action", "reward", "next_state", "done"],
                          defaults=tuple([list() for _ in range(5)]))

        self.memory_conf = {
            'prefix': 'memory',
            'N': 5,
            'max_n_episodes': 3000,
            'spec': spec,
            'gamma': 1.0,
            'max_traj_len': 30
        }

    def __call__(self):
        return super(NstepInputMemoryConfig, self).__call__('spec')


class NstepInputMemory(EpisodicMemory):

    def __init__(self, N, max_n_episodes, spec, gamma, max_traj_len):
        super(NstepInputMemory, self).__init__(max_n_episodes=max_n_episodes,
                                               spec=spec,
                                               gamma=gamma,
                                               max_traj_len=max_traj_len)

        self._cur_traj = Trajectory(gamma=self.gamma, max_len=max_traj_len)
        self.N = N

    def push(self, sample):
        done = sample.done
        self._cur_traj.push(sample)
        if done:
            self.trajectories.append(self._cur_traj)
            self._cur_traj = Trajectory(gamma=self.gamma, max_len=self.max_traj_len)

    def push_trajectories(self, trajectories):
        for trajectory in trajectories:
            self.trajectories.append(trajectory)

    def sample_from_trajectory(self, trajectory_i, sampling_index):
        traj = self.trajectories[trajectory_i]
        i = sampling_index

        hist = []
        for j in range(i - self.N, i):
            state = traj[j].state
            hist.append(state)

        next_hist = []
        for j in range(i - self.N + 1, i + 1):
            state = traj[j].state
            next_hist.append(state)

        state, action, reward, next_state, done = traj[i]
        return hist, state, action, reward, next_hist, next_state, done

    def sample(self, sample_size):
        len_trajectories = self.len_trajectories()
        num_samples_par_trajs = np.clip(len_trajectories - self.N, a_min=0, a_max=np.inf)
        num_samples_par_trajs = num_samples_par_trajs.astype(int)
        effective_num_samples = np.cumsum(num_samples_par_trajs)[-1]

        if sample_size >= effective_num_samples:
            sample_size = effective_num_samples

        p = num_samples_par_trajs / np.sum(num_samples_par_trajs)
        samples_per_traj = np.random.multinomial(sample_size, p)

        hists = []
        states = []
        actions = []
        rewards = []
        next_hists = []
        next_states = []
        dones = []

        for traj_i, num_samples in enumerate(samples_per_traj):
            cur_traj = self.trajectories[traj_i]
            sample_is = np.random.choice(np.arange(self.N, cur_traj.length), num_samples)
            for sample_i in sample_is:
                hs, s, a, r, nhs, ns, d = self.sample_from_trajectory(trajectory_i=traj_i,
                                                                      sampling_index=sample_i)

                hists.append(hs)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_hists.append(nhs)
                next_states.append(ns)
                dones.append(d)

        return hists, states, actions, rewards, next_hists, next_states, dones
