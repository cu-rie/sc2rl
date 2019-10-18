import numpy as np

from sc2rl.runners.RunnerBase import RunnerBase
from sc2rl.utils.HistoryManagers import HistoryManager
from sc2rl.memory.trajectory import Trajectory

from sc2rl.config.graph_configs import NODE_ALLY, NODE_ENEMY
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type


class MultiStepActorRunner(RunnerBase):

    def __init__(self, env, agent, sample_spec, n_steps):
        super(MultiStepActorRunner, self).__init__(env=env, agent=agent, sample_spec=sample_spec)
        self.history_manager = HistoryManager(n_hist_steps=n_steps, init_graph=None)

    def run_1_episode(self):
        trajectory = Trajectory(spec=self.sample_spec, gamma=1.0)
        # the first frame of each episode
        curr_state_dict = self.env.observe()
        curr_graph = curr_state_dict['g']
        self.history_manager.reset(curr_graph)

        while True:
            curr_state_dict = self.env.observe()
            curr_graph = curr_state_dict['g']

            tag2unit_dict = curr_state_dict['tag2unit_dict']
            hist_graph = self.history_manager.get_hist()

            nn_action, sc2_action = self.agent.get_action(hist_graph=hist_graph, curr_graph=curr_graph,
                                                          tag2unit_dict=tag2unit_dict)

            next_state_dict, reward, done = self.env.step(sc2_action)
            next_graph = next_state_dict['g']
            experience = self.sample_spec(curr_graph, nn_action, reward, next_graph, done)

            trajectory.push(experience)
            self.history_manager.append(next_graph)
            if done:
                break

        return Trajectory

    def eval_1_episode(self):
        # expected return
        # dictionary = {'name': (str), 'win': (bool), 'sum_reward': (float)}

        env_name = self.env.name
        traj = self.run_1_episode()

        last_graph = traj[-1].state
        num_allies = get_filtered_node_index_by_type(last_graph, NODE_ALLY)
        num_enemies = get_filtered_node_index_by_type(last_graph, NODE_ENEMY)

        sum_reward = np.sum([exp.reward for exp in traj._trajectory])

        eval_dict = dict()
        eval_dict['name'] = env_name
        eval_dict['win'] = num_allies > num_enemies
        eval_dict['sum_reward'] = sum_reward

        return eval_dict