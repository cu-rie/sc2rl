import os
from functools import partial

import torch
import wandb
import numpy as np

from time import time

import sys

sys.path.append("..")

from sc2rl.utils.reward_funcs import great_victor, great_victor_with_kill_bonus, victory, victory_if_zero_enemy
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl

from sc2rl.rl.agents.Qmix.HierarchicalqmixAgent import HierarchicalQmixAgent, HierarchicalQmixAgentConf
from sc2rl.rl.brains.QMix.HierarchicalqmixBrain import HierarchicalQmixBrainConfig

from sc2rl.rl.networks.MultiStepInputGraphNetwork import MultiStepInputGraphNetworkConfig
from sc2rl.rl.networks.MultiStepInputNetwork import MultiStepInputNetworkConfig
from sc2rl.rl.networks.FeedForward import FeedForwardConfig
from sc2rl.rl.modules.HierarchicalMultiStepQnet import HierarchicalMultiStepInputQnetConfig
from sc2rl.rl.networks.RelationalGraphNetwork import RelationalGraphNetworkConfig
from sc2rl.rl.networks.RelationalNetwork import RelationalNetworkConfig
from sc2rl.rl.brains.QMix.mixer import SupQmixerConf

from sc2rl.rl.modules.MultiStepInputQnet import MultiStepInputQnetConfig
from sc2rl.rl.brains.QMix.qmixBrain import QmixBrainConfig
from sc2rl.rl.agents.Qmix.qmixAgent import QmixAgent, QmixAgentConf

from sc2rl.memory.n_step_memory import NstepInputMemoryConfig
from sc2rl.runners.RunnerManager import RunnerConfig, RunnerManager

from sc2rl.config.graph_configs import (NODE_ALLY, NODE_ENEMY,
                                        EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY,
                                        EDGE_IN_ATTACK_RANGE)

from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment
from sc2rl.utils.HistoryManagers import HistoryManager
from sc2rl.memory.Trajectory import Trajectory

import matplotlib.pyplot as plt

# plt.axis("off")

move_delta = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]) * 0.5


def plot_curr_state(ally_loc, enemy_loc, nn_action, assignment, ax, x_min=20, x_max=45, y_min=20, y_max=36):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ally_colors = ['forestgreen', 'dodgerblue', 'tomato'] # ally colors per group
    enemy_color = 'black'

    # plot enemy loc
    ax.scatter(enemy_loc[:, 0], enemy_loc[:, 1], color=enemy_color, marker='x', s=70)
    for ally in range(len(nn_action)):
        assign = assignment[ally]
        color = ally_colors[assign]
        action = nn_action[ally]
        ally_location = ally_loc[ally]
        ax.scatter(ally_location[0], ally_location[1], color=color)
        if action <= 3:
            move_dir = move_delta[action, :]
            action_location = ally_location + move_dir
            loc = np.stack([ally_location, action_location], axis=0)
        else:
            enemy_location = enemy_loc[action - 5]
            loc = np.stack([ally_location, enemy_location], axis=0)

        ax.plot(loc[:, 0], loc[:, 1], color=color)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    return ax

def plot_curr_state_action(curr_graph, info_dict, nn_action):
    ally_tags = info_dict['ally_tags']
    enemy_tags = info_dict['enemy_tags'][0]
    assignment = info_dict['assignment']

    num_allies = len(ally_tags)
    num_enemies = len(enemy_tags)

    allies = range(num_allies)
    enemies = range(num_allies, num_enemies + num_allies)

    init_node_feature = curr_graph.ndata['init_node_feature']
    agent_loc = init_node_feature[:, 11:13]

    ally_loc = agent_loc[allies, :].numpy()
    enemy_loc = agent_loc[enemies, :].numpy()

    fig, ax = plt.subplots()
    ax = plot_curr_state(ally_loc, enemy_loc, nn_action, assignment, ax)


if __name__ == "__main__":

    # experiment variables
    exp_name = '[NOISY NET] SUB-Q 2Layer ENC'

    use_subq = True

    use_hold = False
    use_tanh = False

    num_hist_time_steps = 2

    victory_coeff = 1.0
    reward_bias = 0.0

    auto_grad_norm_clip = True

    frame_skip_rate = 2
    gamma = 0.95

    eps_init = 0.5
    eps_gamma = 0.997
    tau = 0.1

    gnn_node_update_types = [NODE_ALLY, NODE_ENEMY]
    gnn_edge_update_types = [EDGE_ALLY, EDGE_ENEMY, EDGE_ALLY_TO_ENEMY]

    mixer_node_update_types = [NODE_ALLY]
    mixer_edge_update_types = [EDGE_ALLY]

    use_multi_node_types = True
    exploration_method = 'noisy_net'

    if exploration_method == 'clustered_random':
        eps = eps_init
        use_noisy = False
    elif exploration_method == 'noisy_net':
        eps = 0.0
        use_noisy = True
    else:
        use_noisy = False

    use_absolute_pos = True
    soft_assignment = True
    use_concat_input = False
    use_concat_input_gnn = False
    num_neurons = [64, 64]
    rnn_hidden_size = 32
    use_mixer_hidden = True
    batch_size = 256

    use_attention = False
    use_hypernet = False

    edge_ally_to_enemy = True
    if edge_ally_to_enemy:
        num_relations = 4
    else:
        num_relations = 3

    mixer_num_layer = 1
    enc_gnn_num_layer = 2

    if use_absolute_pos:
        node_input_dim = 19
    else:
        node_input_dim = 17

    if use_mixer_hidden:
        mixer_input_dim = node_input_dim + rnn_hidden_size
    else:
        mixer_input_dim = node_input_dim

    attack_edge_type_index = EDGE_ENEMY

    mixer_rectifier = 'softplus'
    pooling_op = 'softmax'
    pooling_init = None

    map_name = "training_scenario_5"
    spectral_norm = False
    test = False

    num_attn_head = 4
    use_hierarchical_actor = True
    use_double_q = False
    clipped_q = True

    num_runners = 1
    num_samples = 1
    eval_episodes = 11

    # num_runners = 1
    # num_samples = 2
    # eval_episodes = 10

    reward_name = 'victory_if_zero_enemy'

    if use_subq:
        qnet_conf = HierarchicalMultiStepInputQnetConfig(
            multi_step_input_qnet_conf={'exploration_method': exploration_method,
                                        'eps': eps,
                                        'use_attention': use_attention},
            qnet_actor_conf={'spectral_norm': spectral_norm,
                             'node_input_dim': node_input_dim,
                             'pooling_op': pooling_op,
                             'use_concat_input': use_concat_input,
                             'init_node_dim': node_input_dim,
                             'pooling_init': pooling_init,
                             'num_neurons': num_neurons,
                             'attack_edge_type_index': attack_edge_type_index,
                             'use_hold': use_hold,
                             'use_tanh': use_tanh,
                             'use_noisy': use_noisy},
            mixer_conf={'rectifier': mixer_rectifier,
                        'use_attention': use_attention}
        )
    else:
        qnet_conf = MultiStepInputQnetConfig(
            multi_step_input_qnet_conf={'exploration_method': exploration_method,
                                        'eps': eps,
                                        'use_attention': use_attention},
            qnet_actor_conf={'spectral_norm': spectral_norm,
                             'node_input_dim': node_input_dim,
                             'pooling_op': pooling_op,
                             'use_concat_input': use_concat_input,
                             'init_node_dim': node_input_dim,
                             'pooling_init': pooling_init,
                             'num_neurons': num_neurons,
                             'attack_edge_type_index': attack_edge_type_index,
                             'use_hold': use_hold,
                             'use_tanh': use_tanh,
                             'use_noisy': use_noisy}
        )
    if use_attention:
        if use_hypernet:
            gnn_conf = MultiStepInputNetworkConfig(hist_rnn_conf={'input_size': node_input_dim,
                                                                  'hidden_size': rnn_hidden_size},
                                                   hist_enc_conf={'num_layers': enc_gnn_num_layer,
                                                                  'model_dim': node_input_dim,
                                                                  'use_hypernet': use_hypernet,
                                                                  'hypernet_input_dim': num_relations,
                                                                  'num_relations': None,
                                                                  'num_neurons': num_neurons,
                                                                  'num_head': num_attn_head},
                                                   curr_enc_conf={'num_layers': enc_gnn_num_layer,
                                                                  'model_dim': node_input_dim,
                                                                  'use_hypernet': use_hypernet,
                                                                  'hypernet_input_dim': num_relations,
                                                                  'num_relations': None,
                                                                  'num_neurons': num_neurons,
                                                                  'num_head': num_attn_head})
        else:
            gnn_conf = MultiStepInputNetworkConfig(hist_rnn_conf={'input_size': node_input_dim,
                                                                  'hidden_size': rnn_hidden_size},
                                                   hist_enc_conf={'num_layers': enc_gnn_num_layer,
                                                                  'model_dim': node_input_dim,
                                                                  'num_neurons': num_neurons,
                                                                  'num_relations': num_relations,
                                                                  'num_head': num_attn_head},
                                                   curr_enc_conf={'num_layers': enc_gnn_num_layer,
                                                                  'model_dim': node_input_dim,
                                                                  'num_neurons': num_neurons,
                                                                  'num_relations': num_relations,
                                                                  'num_head': num_attn_head})
    else:
        gnn_conf = MultiStepInputGraphNetworkConfig(hist_rnn_conf={'input_size': node_input_dim},
                                                    hist_enc_conf={'spectral_norm': spectral_norm,
                                                                   'num_layers': enc_gnn_num_layer,
                                                                   'model_dim': node_input_dim,
                                                                   'use_concat': use_concat_input_gnn,
                                                                   'num_neurons': num_neurons,
                                                                   'num_relations': num_relations,
                                                                   'use_multi_node_types': use_multi_node_types,
                                                                   'node_update_types': gnn_node_update_types,
                                                                   'edge_update_types': gnn_edge_update_types},
                                                    curr_enc_conf={'spectral_norm': spectral_norm,
                                                                   'num_layers': enc_gnn_num_layer,
                                                                   'model_dim': node_input_dim,
                                                                   'use_concat': use_concat_input_gnn,
                                                                   'num_neurons': num_neurons,
                                                                   'num_relations': num_relations,
                                                                   'use_multi_node_types': use_multi_node_types,
                                                                   'node_update_types': gnn_node_update_types,
                                                                   'edge_update_types': gnn_edge_update_types})
    qnet_conf.gnn_conf = gnn_conf

    buffer_conf = NstepInputMemoryConfig(memory_conf={'use_return': True,
                                                      'N': num_hist_time_steps,
                                                      'gamma': gamma})

    if use_subq:
        brain_conf = HierarchicalQmixBrainConfig(brain_conf={'use_double_q': use_double_q,
                                                             'gamma': gamma,
                                                             'eps': eps_init,
                                                             'eps_gamma': eps_gamma,
                                                             'use_mixer_hidden': use_mixer_hidden},
                                                 fit_conf={'tau': tau,
                                                           'auto_norm_clip': auto_grad_norm_clip})
    else:
        brain_conf = QmixBrainConfig(brain_conf={'use_double_q': use_double_q,
                                                 'gamma': gamma,
                                                 'eps': eps_init,
                                                 'eps_gamma': eps_gamma,
                                                 'use_mixer_hidden': use_mixer_hidden},
                                     fit_conf={'tau': tau,
                                               'auto_norm_clip': auto_grad_norm_clip})

    sample_spec = buffer_conf.memory_conf['spec']
    num_hist_steps = buffer_conf.memory_conf['N']

    run_device = 'cpu'
    fit_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_attention:
        mixer_gnn_conf = RelationalNetworkConfig(gnn_conf={'model_dim': mixer_input_dim,
                                                           'num_layers': mixer_num_layer,
                                                           'num_neurons': num_neurons,
                                                           'num_relations': num_relations,
                                                           'num_head': num_attn_head,
                                                           'use_multi_node_types': False})
    else:
        mixer_gnn_conf = RelationalGraphNetworkConfig(gnn_conf={'spectral_norm': spectral_norm,
                                                                'model_dim': mixer_input_dim,
                                                                'num_layers': mixer_num_layer,
                                                                'use_concat': use_concat_input_gnn,
                                                                'num_neurons': num_neurons,
                                                                'use_multi_node_types': False,
                                                                'node_update_types': mixer_node_update_types,
                                                                'edge_update_types': mixer_edge_update_types,
                                                                'use_noisy': use_noisy})

    mixer_ff_conf = FeedForwardConfig(mlp_conf={'spectral_norm': spectral_norm,
                                                'input_dimension': mixer_input_dim,
                                                'num_neurons': num_neurons,
                                                'use_noisy': use_noisy})

    sup_mixer_conf = SupQmixerConf(nn_conf={'spectral_norm': spectral_norm,
                                            'input_dimension': mixer_input_dim,
                                            'num_neurons': num_neurons,
                                            'use_noisy': use_noisy},
                                   mixer_conf={'rectifier': mixer_rectifier})
    if use_subq:

        agent_conf = HierarchicalQmixAgentConf(agent_conf={'use_clipped_q': clipped_q},
                                               fit_conf={'hist_num_time_steps': num_hist_time_steps,
                                                         'batch_size': batch_size})

        agent = HierarchicalQmixAgent(conf=agent_conf,
                                      qnet_conf=qnet_conf,
                                      mixer_gnn_conf=mixer_gnn_conf,
                                      mixer_ff_conf=mixer_ff_conf,
                                      sup_mixer_conf=sup_mixer_conf,
                                      brain_conf=brain_conf,
                                      buffer_conf=buffer_conf,
                                      soft_assignment=soft_assignment)
    else:
        agent_conf = QmixAgentConf(agent_conf={'use_clipped_q': clipped_q},
                                   fit_conf={'hist_num_time_steps': num_hist_time_steps,
                                             'batch_size': batch_size})

        mixer_gnn_conf.gnn_conf['model_dim'] = mixer_input_dim
        agent = QmixAgent(conf=agent_conf,
                          qnet_conf=qnet_conf,
                          mixer_gnn_conf=mixer_gnn_conf,
                          mixer_ff_conf=mixer_ff_conf,
                          brain_conf=brain_conf,
                          buffer_conf=buffer_conf)

    if exploration_method == 'noisy_net':
        agent.sample_noise()

    agent.to(run_device)

    if reward_name == 'great_victory':
        reward_func = great_victor
    elif reward_name == 'great_victor_with_kill_bonus':
        reward_func = great_victor_with_kill_bonus
    elif reward_name == 'victory':
        reward_func = victory
    elif reward_name == 'victory_if_zero_enemy':
        reward_func = partial(victory_if_zero_enemy, victory_coeff=victory_coeff, reward_bias=reward_bias)
    else:
        raise NotImplementedError("Not supported reward function:{}".format(reward_name))

    game_state_to_dgl = partial(process_game_state_to_dgl,
                                use_absolute_pos=use_absolute_pos,
                                edge_ally_to_enemy=edge_ally_to_enemy)

    config = RunnerConfig(map_name=map_name,
                          reward_func=reward_func,
                          state_proc_func=game_state_to_dgl,
                          agent=agent,
                          n_hist_steps=num_hist_steps,
                          gamma=gamma,
                          realtime=False,
                          frame_skip_rate=frame_skip_rate)

    env = MicroTestEnvironment(map_name=map_name,
                               reward_func=reward_func,
                               state_proc_func=game_state_to_dgl,
                               frame_skip_rate=frame_skip_rate)

    history_manager = HistoryManager(n_hist_steps=num_hist_steps, init_graph=None)

    for i in range(num_samples):

        trajectory = Trajectory(gamma=gamma)
        # the first frame of each episode
        curr_state_dict = env.observe()
        curr_graph = curr_state_dict['g']
        history_manager.reset(curr_graph)

        while True:
            curr_state_dict = env.observe()
            curr_graph = curr_state_dict['g']

            tag2unit_dict = curr_state_dict['tag2unit_dict']
            hist_graph = history_manager.get_hist()

            nn_action, sc2_action, info_dict = agent.get_action(hist_graph=hist_graph, curr_graph=curr_graph,
                                                                tag2unit_dict=tag2unit_dict)

            next_state_dict, reward, done = env.step(sc2_action)
            next_graph = next_state_dict['g']
            experience = sample_spec(curr_graph, nn_action, reward, next_graph, done)

            plot_curr_state_action(curr_graph, info_dict, nn_action)

            trajectory.push(experience)
            history_manager.append(next_graph)

            if done:
                break
