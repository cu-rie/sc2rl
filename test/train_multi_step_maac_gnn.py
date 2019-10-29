import os
import torch
import wandb
import numpy as np

from time import time

from sc2rl.utils.reward_funcs import great_victor_with_kill_bonus
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl

from sc2rl.rl.agents.MAAC.MultiStepActorCriticAgent import MultiStepActorCriticAgent, MultiStepActorCriticAgentConfig
from sc2rl.rl.brains.MAAC.MultiStepActorCriticBrain import MultiStepActorCriticBrainConfig
from sc2rl.rl.networks.MultiStepInputGraphNetwork import MultiStepInputGraphNetworkConfig

from sc2rl.memory.n_step_memory import NstepInputMemoryConfig
from sc2rl.runners.RunnerManager import RunnerConfig, RunnerManager

if __name__ == "__main__":

    map_name = "test_scenario_aligned_aligned_time_adjusted"

    agent_conf = MultiStepActorCriticAgentConfig()
    network_conf = MultiStepInputGraphNetworkConfig()
    brain_conf = MultiStepActorCriticBrainConfig()
    buffer_conf = NstepInputMemoryConfig()
    use_attention = False
    use_hierarchical_actor = True
    num_runners = 1
    num_samples = 2

    sample_spec = buffer_conf.memory_conf['spec']
    num_hist_steps = buffer_conf.memory_conf['N']

    run_device = 'cpu'
    fit_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = MultiStepActorCriticAgent(agent_conf,
                                      network_conf,
                                      brain_conf,
                                      buffer_conf,
                                      use_attention=use_attention,
                                      use_hierarchical_actor=use_hierarchical_actor)
    agent.to(run_device)

    config = RunnerConfig(map_name=map_name,
                          reward_func=great_victor_with_kill_bonus,
                          state_proc_func=process_game_state_to_dgl,
                          agent=agent,
                          n_hist_steps=num_hist_steps)

    runner_manager = RunnerManager(config, num_runners)

    wandb.init(project="sc2rl")
    wandb.watch(agent)
    wandb.config.update({'use_attention': use_attention,
                         'num_runners': num_runners,
                         'num_samples': num_samples,
                         'use_hierarchical_actor': use_hierarchical_actor,
                         'map_name': map_name})
    wandb.config.update(agent_conf())
    wandb.config.update(network_conf())
    wandb.config.update(brain_conf())
    wandb.config.update(buffer_conf())

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
            print("fit time : {}".format(e_time-s_time))

            wandb.log(fit_return_dict, step=iters)
            wrs = [runner.env.winning_ratio for runner in runner_manager.runners]
            mean_wr = np.mean(wrs)
            wandb.log(fit_return_dict, step=iters)
            wandb.log({'winning_ratio': mean_wr}, step=iters)

            if iters % 20 == 0:
                save_path = os.path.join(os.getcwd(), 'exp_{}_{}.ptb'.format(iters, map_name))
                torch.save(agent.state_dict(), save_path)

        runner_manager.close()
    except KeyboardInterrupt:
        runner_manager.close()
    finally:
        runner_manager.close()
