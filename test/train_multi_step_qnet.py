import os
import torch
import wandb
import numpy as np

import context

from time import time

from sc2rl.utils.reward_funcs import great_victor, great_victor_with_kill_bonus, victory, victory_if_zero_enemy
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl

from sc2rl.rl.brains.QMix.qmixBrain import QmixBrainConfig
from sc2rl.rl.agents.Qmix.qmixAgent import QmixAgent, QmixAgentConf
from sc2rl.rl.modules.MultiStepInputQnet import MultiStepInputQnetConfig
from sc2rl.rl.networks.MultiStepInputGraphNetwork import MultiStepInputGraphNetworkConfig
from sc2rl.rl.networks.MultiStepInputNetwork import MultiStepInputNetworkConfig
from sc2rl.rl.networks.FeedForward import FeedForwardConfig
from sc2rl.rl.networks.RelationalGraphNetwork import RelationalGraphNetworkConfig

from sc2rl.memory.n_step_memory import NstepInputMemoryConfig
from sc2rl.runners.RunnerManager import RunnerConfig, RunnerManager

if __name__ == "__main__":

    map_name = "training_scenario_4"
    spectral_norm = False

    use_attention = False
    use_hierarchical_actor = True
    num_runners = 2
    num_samples = 10
    eval_episodes = 20
    reward_name = 'victory_if_zero_enemy'
    exp_name = "[S4] scheduler"

    qnet_conf = MultiStepInputQnetConfig(multi_step_input_qnet_conf={'exploration_method': 'clustered_random'},
                                         qnet_actor_conf={'spectral_norm': spectral_norm})
    if use_attention:
        gnn_conf = MultiStepInputNetworkConfig()
    else:
        gnn_conf = MultiStepInputGraphNetworkConfig(hist_enc_conf={'spectral_norm': spectral_norm},
                                                    curr_enc_conf={'spectral_norm': spectral_norm})

    qnet_conf.gnn_conf = gnn_conf

    buffer_conf = NstepInputMemoryConfig(memory_conf={'use_return': True})
    brain_conf = QmixBrainConfig(brain_conf={'use_double_q': True},
                                 fit_conf={'tau': 0.9})

    sample_spec = buffer_conf.memory_conf['spec']
    num_hist_steps = buffer_conf.memory_conf['N']

    run_device = 'cpu'
    fit_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_attention:
        raise NotImplementedError
    else:
        mixer_gnn_conf = RelationalGraphNetworkConfig(gnn_conf={'spectral_norm': spectral_norm})
    mixer_ff_conf = FeedForwardConfig(mlp_conf={'spectral_norm': spectral_norm})

    agent_conf = QmixAgentConf(agent_conf={'use_clipped_q': True})

    agent = QmixAgent(conf=agent_conf,
                      qnet_conf=qnet_conf,
                      mixer_gnn_conf=mixer_gnn_conf,
                      mixer_ff_conf=mixer_ff_conf,
                      brain_conf=brain_conf,
                      buffer_conf=buffer_conf)

    agent.to(run_device)

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

    config = RunnerConfig(map_name=map_name,
                          reward_func=reward_func,
                          state_proc_func=process_game_state_to_dgl,
                          agent=agent,
                          n_hist_steps=num_hist_steps)

    runner_manager = RunnerManager(config, num_runners)

    wandb.init(project="qmix", name=exp_name)
    wandb.watch(agent)
    wandb.config.update({'use_attention': use_attention,
                         'num_runners': num_runners,
                         'num_samples': num_samples,
                         'use_hierarchical_actor': use_hierarchical_actor,
                         'map_name': map_name,
                         'reward': reward_name})
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
