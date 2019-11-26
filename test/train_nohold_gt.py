import os
from functools import partial

import torch
import wandb
import numpy as np

import context

from time import time

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

from sc2rl.memory.n_step_memory import NstepInputMemoryConfig
from sc2rl.runners.RunnerManager import RunnerConfig, RunnerManager

from sc2rl.config.graph_configs import EDGE_IN_ATTACK_RANGE, EDGE_ENEMY

if __name__ == "__main__":

    # experiment variables
    exp_name = '[RETRAIN] HOPE NEW GT - NO HOLD'

    use_hold = False

    num_hist_time_steps = 2

    victory_coeff = 1.0
    reward_bias = 2.0

    auto_grad_norm_clip = True

    frame_skip_rate = 2
    gamma = 1.0

    eps_init = 0.5
    eps_gamma = 0.997
    tau = 0.1

    use_absolute_pos = True
    soft_assignment = True
    use_concat_input = True
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
    enc_gnn_num_layer = 1

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
    pooling_op = 'relu'
    pooling_init = None

    map_name = "training_scenario_4"
    spectral_norm = False
    test = False

    num_attn_head = 4
    use_hierarchical_actor = True
    use_double_q = True
    clipped_q = False

    num_runners = 1
    num_samples = 4
    eval_episodes = 1

    # num_runners = 2
    # num_samples = 4
    # eval_episodes = 10

    reward_name = 'victory_if_zero_enemy'

    qnet_conf = HierarchicalMultiStepInputQnetConfig(
        multi_step_input_qnet_conf={'exploration_method': 'clustered_random',
                                    'use_attention': use_attention},
        qnet_actor_conf={'spectral_norm': spectral_norm,
                         'node_input_dim': node_input_dim,
                         'pooling_op': pooling_op,
                         'use_concat_input': use_concat_input,
                         'init_node_dim': node_input_dim,
                         'pooling_init': pooling_init,
                         'num_neurons': num_neurons,
                         'attack_edge_type_index': attack_edge_type_index,
                         'use_hold': use_hold},
        mixer_conf={'rectifier': mixer_rectifier,
                    'use_attention': use_attention}
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
                                                                   'num_relations': num_relations},
                                                    curr_enc_conf={'spectral_norm': spectral_norm,
                                                                   'num_layers': enc_gnn_num_layer,
                                                                   'model_dim': node_input_dim,
                                                                   'use_concat': use_concat_input_gnn,
                                                                   'num_neurons': num_neurons,
                                                                   'num_relations': num_relations})
    qnet_conf.gnn_conf = gnn_conf

    buffer_conf = NstepInputMemoryConfig(memory_conf={'use_return': True,
                                                      'N': num_hist_time_steps,
                                                      'gamma': gamma})
    brain_conf = HierarchicalQmixBrainConfig(brain_conf={'use_double_q': use_double_q,
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
                                                           'num_head': num_attn_head})
    else:
        mixer_gnn_conf = RelationalGraphNetworkConfig(gnn_conf={'spectral_norm': spectral_norm,
                                                                'model_dim': mixer_input_dim,
                                                                'num_layers': mixer_num_layer,
                                                                'use_concat': use_concat_input_gnn,
                                                                'num_neurons': num_neurons})

    mixer_ff_conf = FeedForwardConfig(mlp_conf={'spectral_norm': spectral_norm,
                                                'input_dimension': mixer_input_dim,
                                                'num_neurons': num_neurons})

    sup_mixer_conf = SupQmixerConf(nn_conf={'spectral_norm': spectral_norm,
                                            'input_dimension': mixer_input_dim,
                                            'num_neurons': num_neurons},
                                   mixer_conf={'rectifier': mixer_rectifier})

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

    agent.to(run_device)

    if test:
        # if use_absolute_pos:
        #     load_path = 'abs_pos.ptb'
        # else:
        #     load_path = 'no_abs_pos.ptb'
        load_path = '1800.ptb'
        agent.load_state_dict(torch.load(load_path))

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
                          realtime=False)

    runner_manager = RunnerManager(config, num_runners)

    wandb.init(project="qmix2", name=exp_name)
    wandb.watch(agent)
    wandb.config.update({'use_attention': use_attention,
                         'num_runners': num_runners,
                         'num_samples': num_samples,
                         'use_hierarchical_actor': use_hierarchical_actor,
                         'map_name': map_name,
                         'reward': reward_name,
                         'frame_skip_rate': frame_skip_rate,
                         'use_absolute_pos': use_absolute_pos,
                         'victory_coeff': victory_coeff})

    wandb.config.update(agent_conf())
    wandb.config.update(gnn_conf())
    wandb.config.update(brain_conf())
    wandb.config.update(buffer_conf())
    wandb.config.update(qnet_conf())
    wandb.config.update(mixer_gnn_conf())
    wandb.config.update(mixer_ff_conf())

    try:
        iters = 0
        while iters < 1000000:
            iters += 1
            runner_manager.sample(num_samples)
            runner_manager.transfer_sample()

            s_time = time()
            agent.to(fit_device)
            fit_return_dict = agent.fit(device=fit_device)
            agent.to(run_device)
            e_time = time()

            running_wrs = [runner.env.winning_ratio for runner in runner_manager.runners]
            running_wr = np.mean(running_wrs)
            wandb.log(fit_return_dict, step=iters)
            wandb.log({'train_winning_ratio': running_wr, 'epsilon': agent.brain.eps}, step=iters)

            if iters % 20 == 0:
                save_path = os.path.join(wandb.run.dir, '{}.ptb'.format(iters))
                torch.save(agent.state_dict(), save_path)

            if iters % 5 == 0:
                eval_dicts = runner_manager.evaluate(eval_episodes)
                wins = []
                for eval_dict in eval_dicts:
                    win = eval_dict['win']
                    wins.append(win)

                wr = np.mean(np.array(wins))
                wandb.log({'eval_winning_ratio': wr}, step=iters)

        runner_manager.close()
    except KeyboardInterrupt:
        runner_manager.close()
    finally:
        runner_manager.close()
