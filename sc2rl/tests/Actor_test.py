from functools import partial

import dgl
import torch

from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl
from sc2rl.utils.graph_utils import pop_node_feature_dict

from sc2rl.rl.networks.rnn_encoder import RNNEncoder
from sc2rl.nn.RelationalNetwork import RelationalNetwork
from sc2rl.rl.networks.Actor import Actor


def reward_func(s, ns):
    return 1


def filter_by_edge_type_idx(edges, etype_idx):
    return edges.data['edge_type'] == etype_idx


if __name__ == "__main__":
    map_name = "2m_vs_1hellion"
    test_reward_func = reward_func
    test_sate_proc_func = process_game_state_to_dgl
    done_cnt = 0

    env = MicroTestEnvironment(map_name, test_reward_func, test_sate_proc_func)
    rnn = torch.nn.GRU(input_size=10,
                       hidden_size=10,
                       num_layers=2,
                       batch_first=True)

    hist_enc = RelationalNetwork(num_layers=1,
                                 model_dim=20,
                                 use_hypernet=False,
                                 hypernet_input_dim=None,
                                 num_relations=3,
                                 num_head=1,
                                 use_norm=True,
                                 neighbor_degree=0,
                                 num_neurons=[128, 128],
                                 pooling_op='relu')

    global_encoder = RNNEncoder(rnn=rnn, one_step_encoder=hist_enc)
    actor = Actor(global_encoder=global_encoder,
                  node_dim=20

                  )

    while True:
        cur_state_dict = env.observe()
        cur_state = cur_state_dict['g']

        cur_state = dgl.batch([cur_state, cur_state])
        filter_func = partial(filter_by_edge_type_idx, etype_idx=1)
        filter_edge_idx = cur_state.filter_edges(filter_func)

        cur_state_feature_dict = pop_node_feature_dict(cur_state)

        action = actor(batch_size=1, num_time_steps=1,
                       batch_time_batched_graph=cur_state,
                       feature_dict=cur_state_feature_dict)
        next_state, reward, done = env.step(action=None)
        if done:
            done_cnt += 1
            if done_cnt >= 10:
                break

    env.close()

    print("We are in the end game.")
