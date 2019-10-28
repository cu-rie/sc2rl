import dgl
from collections import deque

from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment
from sc2rl.utils.reward_funcs import great_victor_with_kill_bonus
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl

from sc2rl.rl.agents.MAAC.MultiStepActorCriticAgent import MultiStepActorCriticAgent, MultiStepActorCriticAgentConfig
from sc2rl.rl.brains.MAAC.MultiStepInputActorCriticBrain_refac import MultiStepActorCriticBrainConfig
from sc2rl.rl.networks.MultiStepInputNetwork import MultiStepInputNetworkConfig
from sc2rl.memory.n_step_memory import NstepInputMemoryConfig


class HistoryManager:

    def __init__(self, n_hist_steps, init_graph):
        self.n_hist_steps = n_hist_steps
        self.hist = deque(maxlen=n_hist_steps)
        self.reset(init_graph)

    def append(self, graph):
        self.hist.append(graph)

    def get_hist(self):
        return dgl.batch([g for g in self.hist])

    def reset(self, init_graph):
        self.hist.clear()
        for _ in range(self.n_hist_steps):
            self.hist.append(init_graph)


if __name__ == "__main__":
    map_name = "training_scenario_1"
    env = MicroTestEnvironment(map_name=map_name,
                               reward_func=great_victor_with_kill_bonus,
                               state_proc_func=process_game_state_to_dgl)

    agent_conf = MultiStepActorCriticAgentConfig()
    network_conf = MultiStepInputNetworkConfig()
    brain_conf = MultiStepActorCriticBrainConfig()
    buffer_conf = NstepInputMemoryConfig()

    sample_spec = buffer_conf.memory_conf['spec']

    agent = MultiStepActorCriticAgent(agent_conf,
                                      network_conf,
                                      brain_conf,
                                      buffer_conf)

    init_graph = env.observe()['g']
    history_manager = HistoryManager(
        n_hist_steps=buffer_conf.memory_conf['N'], init_graph=init_graph)

    done_cnt = 0
    iters = 0
    while True:
        # print("Itertation : {} ".format(iters))
        curr_state_dict = env.observe()
        hist_graph = history_manager.get_hist()
        curr_graph = curr_state_dict['g']

        tag2unit_dict = curr_state_dict['tag2unit_dict']

        nn_action, sc2_action = agent.get_action(hist_graph=hist_graph, curr_graph=curr_graph,
                                                 tag2unit_dict=tag2unit_dict)

        next_state_dict, reward, done = env.step(sc2_action)
        next_graph = next_state_dict['g']
        experience = sample_spec(
            curr_graph, nn_action, reward, next_graph, done)

        agent.append_sample(experience)
        history_manager.append(next_graph)

        if done:
            done_cnt += 1
            if done_cnt % 1 == 0:
                print("fit at {}".format(done_cnt))
                fit_return_dict = agent.fit()

            if done_cnt >= 1000:
                break
        iters += 1
    env.close()
