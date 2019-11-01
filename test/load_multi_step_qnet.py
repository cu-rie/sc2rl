import os
import torch

import numpy as np

import context

from sc2rl.utils.reward_funcs import great_victor_with_kill_bonus
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl

from sc2rl.rl.brains.QMix.qmixBrain import QmixBrainConfig
from sc2rl.rl.agents.Qmix.qmixAgent import QmixAgent, QmixAgentConf
from sc2rl.rl.modules.MultiStepInputQnet import MultiStepInputQnetConfig
from sc2rl.rl.networks.MultiStepInputGraphNetwork import MultiStepInputGraphNetworkConfig

from sc2rl.memory.n_step_memory import NstepInputMemoryConfig
from sc2rl.runners.RunnerManager import RunnerConfig, RunnerManager

if __name__ == "__main__":

    map_name = "10m_vs_8_11m_all_random"

    agent_conf = QmixAgentConf()
    network_conf = MultiStepInputGraphNetworkConfig()
    brain_conf = QmixBrainConfig()

    qnet_conf = MultiStepInputQnetConfig()
    buffer_conf = NstepInputMemoryConfig()
    use_attention = False
    use_hierarchical_actor = True
    num_runners = 1
    num_samples = 1

    sample_spec = buffer_conf.memory_conf['spec']
    num_hist_steps = buffer_conf.memory_conf['N']

    run_device = 'cpu'
    fit_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = QmixAgent(conf=agent_conf,
                      qnet_conf=qnet_conf,
                      brain_conf=brain_conf,
                      buffer_conf=buffer_conf)

    agent.brain.eps.fill_(0.0001)

    path = os.path.join(os.getcwd(), 'wandb', 'run-20191030_012513-l5r9goml', '300.ptb')
    agent.load_state_dict(torch.load(path))

    agent.to(run_device)

    config = RunnerConfig(map_name=map_name,
                          reward_func=great_victor_with_kill_bonus,
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
