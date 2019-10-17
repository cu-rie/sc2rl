from sc2rl.memory.memory_base import EpisodicMemory
from sc2rl.memory.trajectory import Trajectory

import numpy as np


class NstepInputMemory(EpisodicMemory):

    def __init__(self, N, max_n_episodes, spec, gamma, max_traj_len):
        super(NstepInputMemory, self).__init__(max_n_episodes=max_n_episodes,
                                               spec=spec,
                                               gamma=gamma,
                                               max_traj_len=max_traj_len)

        self.N = N

    def push(self, sample):
        done = sample.done
        self._cur_traj.push(sample)
        if done:
            self.trajectories.append(self._cur_traj)
            self._cur_traj = Trajectory(spec=self.spec, gamma=self.gamma, max_len=self.max_traj_len)

    def sample_from_trajectory(self, trajectory_i, sampling_index):
        traj = self.trajectories[trajectory_i]
        i = sampling_index

        hist = []
        for j in range(i - self.N, i):
            state, _, _, _, _ = traj[j]
            hist.append(state)

        next_hist = []
        for j in range(i - self.N + 1, i + 1):
            state, _, _, _, _ = traj[j]
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
            sample_is = np.random.choice(np.arange(self.N, cur_traj.len_trajectory), num_samples)
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