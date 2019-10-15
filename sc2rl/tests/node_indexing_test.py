from functools import partial

import dgl
import torch

from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment
from sc2rl.utils.state_to_graph import process_game_state_to_dgl
from sc2rl.utils.graph_utils import pop_node_feature_dict

from sc2rl.rl.rl_networks.rnn_encoder import RNNEncoder
from sc2rl.nn.RelationalNetwork import RelationalNetwork
from sc2rl.config.graph_configs import NODE_ALLY, EDGE_ALLY, EDGE_ENEMY, EDGE_IN_ATTACK_RANGE

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
                                 model_dim=26,
                                 use_hypernet=False,
                                 hypernet_input_dim=None,
                                 num_relations=3,
                                 num_head=1,
                                 use_norm=True,
                                 neighbor_degree=1,
                                 num_neurons=[128, 128],
                                 pooling_op='relu')

    actor = RNNEncoder(rnn=rnn, one_step_encoder=hist_enc)

    while True:
        cur_state_dict = env.observe()
        cur_state = cur_state_dict['g']

        cur_state = dgl.batch([cur_state, cur_state])

        cur_state_feature_dict = cur_state.ndata.pop('node_feature')

        enc_out = hist_enc(cur_state, cur_state_feature_dict, [NODE_ALLY], [EDGE_ALLY, EDGE_ENEMY])

        next_state, reward, done = env.step(action=None)
        if done:
            done_cnt += 1
            if done_cnt >= 10:
                break

    env.close()

    print("We are in the end game.")
