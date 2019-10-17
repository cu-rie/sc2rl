import dgl
import torch
from collections import namedtuple, deque

from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment
from sc2rl.utils.reward_funcs import great_victor_with_kill_bonus
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl

from sc2rl.memory.n_step_memory import NstepInputMemory
from sc2rl.rl.networks.RelationalNetwork import RelationalNetwork
from sc2rl.rl.networks.rnn_encoder import RNNEncoder
from sc2rl.rl.agents.MultiStepActorCriticAgent import MultiStepActorCriticAgent
from sc2rl.rl.brains.MultiStepActorCriticBrain import MultiStepActorCriticBrain
from sc2rl.rl.modules.ActorCritic import ActorCriticModule


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

    node_dim = 26
    rnn_hidden_dim = 32
    rnn_layers = 2

    rnn = torch.nn.GRU(input_size=node_dim,
                       hidden_size=rnn_hidden_dim,
                       num_layers=rnn_layers,
                       batch_first=True)

    num_hist_enc_layer = 1

    one_step_hist_enc = RelationalNetwork(num_layers=num_hist_enc_layer,
                                          model_dim=node_dim,
                                          use_hypernet=False,
                                          hypernet_input_dim=None,
                                          num_relations=3,
                                          num_head=2,
                                          use_norm=True,
                                          neighbor_degree=0,
                                          num_neurons=[128, 128],
                                          pooling_op='relu')

    hist_encoder = RNNEncoder(rnn=rnn, one_step_encoder=one_step_hist_enc)

    num_cur_enc_layer = 1

    curr_encoder = RelationalNetwork(num_layers=num_cur_enc_layer,
                                     model_dim=node_dim,
                                     use_hypernet=False,
                                     hypernet_input_dim=None,
                                     num_relations=3,
                                     num_head=2,
                                     use_norm=True,
                                     neighbor_degree=0,
                                     num_neurons=[128, 128],
                                     pooling_op='relu')

    actor_critic = ActorCriticModule(node_input_dim=rnn_hidden_dim + node_dim)

    brain = MultiStepActorCriticBrain(actor_critic=actor_critic, hist_encoder=hist_encoder, curr_encoder=curr_encoder)

    sample_spec = namedtuple('exp_args', ["state", "action", "reward", "next_state", "done"],
                             defaults=tuple([list() for _ in range(4)]))

    num_hist_steps = 5

    buffer = NstepInputMemory(N=num_hist_steps, max_n_episodes=100, spec=sample_spec, gamma=1.0, max_traj_len=40)

    agent = MultiStepActorCriticAgent(brain=brain, buffer=buffer)

    init_graph = env.observe()['g']
    history_manager = HistoryManager(n_hist_steps=num_hist_steps, init_graph=init_graph)

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
        experience = sample_spec(curr_graph, nn_action, reward, next_graph, done)

        agent.append_sample(experience)
        history_manager.append(next_graph)

        if done:
            done_cnt += 1
            if done_cnt % 20 == 0:
                print("fit at {}".format(done_cnt))
                agent.fit(batch_size=20, hist_num_time_steps=num_hist_steps)

            if done_cnt >= 100:
                break
        iters += 1
    env.close()
