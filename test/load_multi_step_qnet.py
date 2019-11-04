import os
import torch
import wandb
import numpy as np

from time import time

from sc2rl.utils.reward_funcs import great_victor, great_victor_with_kill_bonus
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

    map_name = "training_scenario_1"
    spectral_norm = True

    agent_conf = QmixAgentConf()

    use_attention = False
    use_hierarchical_actor = True
    num_runners = 1
    num_samples = 20

    qnet_conf = MultiStepInputQnetConfig(qnet_actor_conf={'spectral_norm': spectral_norm})
    if use_attention:
        gnn_conf = MultiStepInputNetworkConfig()
    else:
        gnn_conf = MultiStepInputGraphNetworkConfig(hist_enc_conf={'spectral_norm': spectral_norm},
                                                    curr_enc_conf={'spectral_norm': spectral_norm})

    qnet_conf.gnn_conf = gnn_conf

    buffer_conf = NstepInputMemoryConfig(memory_conf={'use_return': True})
    brain_conf = QmixBrainConfig()

    sample_spec = buffer_conf.memory_conf['spec']
    num_hist_steps = buffer_conf.memory_conf['N']

    run_device = 'cpu'
    fit_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_attention:
        raise NotImplementedError
    else:
        mixer_gnn_conf = RelationalGraphNetworkConfig(gnn_conf={'spectral_norm': spectral_norm})
    mixer_ff_conf = FeedForwardConfig(mlp_conf={'spectral_norm': spectral_norm})

    agent = QmixAgent(conf=agent_conf,
                      qnet_conf=qnet_conf,
                      mixer_gnn_conf=mixer_gnn_conf,
                      mixer_ff_conf=mixer_ff_conf,
                      brain_conf=brain_conf,
                      buffer_conf=buffer_conf)

    path = os.path.join(os.getcwd(), 'wandb', 'run-20191101_120825-gvzc3jfh', '200.ptb')
    agent.load_state_dict(torch.load(path))

    agent.to(run_device)

    reward_name = 'great_victory'
    if reward_name == 'great_victory':
        reward_func = great_victor
    elif reward_name == 'great_victor_with_kill_bonus':
        reward_func = great_victor_with_kill_bonus

    config = RunnerConfig(map_name=map_name,
                          reward_func=reward_func,
                          state_proc_func=process_game_state_to_dgl,
                          agent=agent,
                          n_hist_steps=num_hist_steps)

    runner_manager = RunnerManager(config, num_runners)

    try:
        iters = 0
        while iters < 1000000:
            iters += 1
            runner_manager.sample(num_samples)
            runner_manager.transfer_sample()
            wrs = [runner.env.winning_ratio for runner in runner_manager.runners]
            mean_wr = np.mean(wrs)
            print("winning ratio : {} ".format(mean_wr))

        runner_manager.close()
    except KeyboardInterrupt:
        runner_manager.close()
    finally:
        runner_manager.close()
