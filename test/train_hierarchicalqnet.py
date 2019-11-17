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
from sc2rl.rl.brains.QMix.mixer import SupQmixerConf

from sc2rl.memory.n_step_memory import NstepInputMemoryConfig
from sc2rl.runners.RunnerManager import RunnerConfig, RunnerManager

if __name__ == "__main__":

    # experiment variables
    exp_name = "DEBUG"

    num_hist_time_steps = 2

    frame_skip_rate = 2
    use_absolute_pos = True
    soft_assignment = True
    use_concat_input = True

    if use_absolute_pos:
        node_input_dim = 19
    else:
        node_input_dim = 17

    mixer_rectifier = 'softplus'
    pooling_op = None

    map_name = "training_scenario_4"
    spectral_norm = False
    test = False

    use_attention = False
    use_hierarchical_actor = True
    use_double_q = True
    clipped_q = False

    num_runners = 2
    num_samples = 20
    eval_episodes = 10
    reward_name = 'victory_if_zero_enemy'

    qnet_conf = HierarchicalMultiStepInputQnetConfig(
        multi_step_input_qnet_conf={'exploration_method': 'clustered_random'},
        qnet_actor_conf={'spectral_norm': spectral_norm,
                         'node_input_dim': node_input_dim,
                         'pooling_op': pooling_op,
                         'use_concat_input': use_concat_input,
                         'init_node_dim': node_input_dim},
        mixer_conf={'rectifier': mixer_rectifier}
    )
    if use_attention:
        gnn_conf = MultiStepInputNetworkConfig()
    else:
        gnn_conf = MultiStepInputGraphNetworkConfig(hist_rnn_conf={'input_size': node_input_dim},
                                                    hist_enc_conf={'spectral_norm': spectral_norm,
                                                                   'model_dim': node_input_dim},
                                                    curr_enc_conf={'spectral_norm': spectral_norm,
                                                                   'model_dim': node_input_dim})
    qnet_conf.gnn_conf = gnn_conf

    buffer_conf = NstepInputMemoryConfig(memory_conf={'use_return': True,
                                                      'N': num_hist_time_steps})
    brain_conf = HierarchicalQmixBrainConfig(brain_conf={'use_double_q': use_double_q},
                                             fit_conf={'tau': 0.9})

    sample_spec = buffer_conf.memory_conf['spec']
    num_hist_steps = buffer_conf.memory_conf['N']

    run_device = 'cpu'
    fit_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_attention:
        raise NotImplementedError
    else:
        mixer_gnn_conf = RelationalGraphNetworkConfig(gnn_conf={'spectral_norm': spectral_norm,
                                                                'model_dim': node_input_dim})
    mixer_ff_conf = FeedForwardConfig(mlp_conf={'spectral_norm': spectral_norm,
                                                'input_dimension': node_input_dim})

    sup_mixer_conf = SupQmixerConf(nn_conf={'spectral_norm': spectral_norm,
                                            'input_dimension': node_input_dim},
                                   mixer_conf={'rectifier': mixer_rectifier})

    agent_conf = HierarchicalQmixAgentConf(agent_conf={'use_clipped_q': clipped_q},
                                           fit_conf={'hist_num_time_steps': num_hist_time_steps})

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
        load_path = '2820.ptb'
        agent.load_state_dict(torch.load(load_path))

    if reward_name == 'great_victory':
        reward_func = great_victor
    elif reward_name == 'great_victor_with_kill_bonus':
        reward_func = great_victor_with_kill_bonus
    elif reward_name == 'victory':
        reward_func = victory
    elif reward_name == 'victory_if_zero_enemy':
        reward_func = victory_if_zero_enemy
    else:
        raise NotImplementedError("Not supported reward function:{}".format(reward_name))

    game_state_to_dgl = partial(process_game_state_to_dgl, use_absolute_pos=use_absolute_pos)
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
                         'use_absolute_pos': use_absolute_pos})

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
