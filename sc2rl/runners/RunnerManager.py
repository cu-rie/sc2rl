from queue import Queue
import threading as thr
from collections import namedtuple

from sc2rl.runners.MultiStepActorRunner import MultiStepActorRunner
from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment


class RunnerConfig:
    def __init__(self, map_name, reward_func, state_proc_func, agent, n_hist_steps, gamma,
                 frame_skip_rate=1, realtime=False):
        self.env_config = {
            "map_name": map_name,
            "reward_func": reward_func,
            "state_proc_func": state_proc_func,
            "frame_skip_rate": frame_skip_rate,
            "realtime": realtime,
        }

        self.agent = agent
        self.n_hist_steps = n_hist_steps
        # python version == 3.7
        # self.sample_spec = namedtuple('exp_args', ["state", "action", "reward", "next_state", "done", "ret"],
        #                               defaults=tuple([list() for _ in range(6)]))
        # python version < 3.7
        self.sample_spec = namedtuple('exp_args', ["state", "action", "reward", "next_state", "done", "ret"])
        self.sample_spec.__new__.__defaults__ = tuple([list() for _ in range(6)])

        self.gamma = gamma


class RunnerManagerBase:
    def __init__(self, config, num_runners):
        raise NotImplementedError(
            "This method will be implemented in the child class")

    def sample(self, total_n):
        raise NotImplementedError(
            "This method will be implemented in the child class")

    def evaluate(self, total_n):
        raise NotImplementedError(
            "This method will be implemented in the child class")


class RunnerManager:
    def __init__(self, config, num_runners):
        self.num_runners = num_runners
        self.agent = config.agent
        self.runners = []

        self.sample_queue = Queue()
        self.eval_queue = Queue()

        for _ in range(num_runners):
            env = MicroTestEnvironment(**config.env_config)

            self.runners.append(MultiStepActorRunner(
                env, config.agent, config.sample_spec, config.n_hist_steps, config.gamma))

    def sample(self, total_n):
        self.reset()

        threads = []
        for (n, runner) in zip(self._calc_n(total_n), self.runners):
            th = thr.Thread(target=runner.run_n_episodes,
                            args=(n, self.sample_queue))
            threads.append(th)

        for th in threads:
            th.start()

        for th in threads:
            th.join()

    def transfer_sample(self):
        trajectories = []
        while not self.sample_queue.empty():
            traj = self.sample_queue.get()
            trajectories.append(traj)

        self.agent.buffer.push_trajectories(trajectories)

    def evaluate(self, total_n):
        self.reset()
        self.set_eval_mode()

        threads = []
        for (n, runner) in zip(self._calc_n(total_n), self.runners):
            th = thr.Thread(target=runner.eval_n_episodes,
                            args=(n, self.eval_queue))
            threads.append(th)

        for th in threads:
            th.start()

        for th in threads:
            th.join()

        eval_dicts = []
        while not self.eval_queue.empty():
            eval_dict = self.eval_queue.get()
            eval_dicts.append(eval_dict)

        self.set_train_mode()
        return eval_dicts

    def set_eval_mode(self):
        for runner in self.runners:
            runner.set_eval_mode()

    def set_train_mode(self):
        for runner in self.runners:
            runner.set_train_mode()

    def close(self):
        for runner in self.runners:
            runner.close()

    def reset(self):
        for runner in self.runners:
            runner.reset()

    def _calc_n(self, total_n):
        div, remain = divmod(total_n, self.num_runners)
        ns = [div] * (self.num_runners - remain) + [div + 1] * remain
        return ns
