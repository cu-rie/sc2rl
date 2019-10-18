import dgl
import torch

# env related
from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl
from sc2rl.utils.reward_funcs import great_victor_with_kill_bonus

from sc2rl.config.graph_configs import NODE_ALLY
from sc2rl.config.graph_configs import EDGE_ALLY, EDGE_ENEMY

# rl related
from sc2rl.rl.networks.rnn_encoder import RNNEncoder
from sc2rl.rl.networks.RelationalNetwork import RelationalNetwork

if __name__ == "__main__":
    map_name = "3m_vs_4m_randoffset"
    env = MicroTestEnvironment(map_name=map_name,
                               reward_func=great_victor_with_kill_bonus,
                               state_proc_func=process_game_state_to_dgl)

    rnn = torch.nn.GRU(input_size=20,
                       hidden_size=32,
                       num_layers=2,
                       batch_first=True)

    one_step_enc = RelationalNetwork(num_layers=1,
                                     model_dim=20,
                                     use_hypernet=False,
                                     hypernet_input_dim=None,
                                     num_relations=3,
                                     num_head=1,
                                     use_norm=True,
                                     neighbor_degree=1,
                                     num_neurons=[128, 128],
                                     pooling_op='relu')

    hist_enc = RNNEncoder(rnn=rnn, one_step_encoder=one_step_enc)

    done_cnt = 0
    while True:
        cur_state_dict = env.observe()
        g = cur_state_dict['g']

        batched_g = dgl.batch([g, g, g, g, g, g])
        node_feat = batched_g.ndata.pop('node_feature')

        hist_out, hist_hidden = hist_enc(num_time_steps=2,
                                         batch_time_batched_graph=batched_g,
                                         node_feat=node_feat)

        next_state, reward, done = env.step(action=None)
        if done:
            done_cnt += 1
            if done_cnt >= 10:
                break
