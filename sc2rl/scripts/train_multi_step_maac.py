import wandb
import torch
import numpy as np
import os

from sc2rl.utils.reward_funcs import great_victor_with_kill_bonus
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl

from sc2rl.rl.agents.MultiStepActorCriticAgent_refac import MultiStepActorCriticAgent, MultiStepActorCriticAgentConfig
from sc2rl.rl.brains.MultiStepInputActorCriticBrain_refac import MultiStepActorCriticBrainConfig
from sc2rl.rl.networks.MultiStepInputNetwork import MultiStepInputNetworkConfig

from sc2rl.memory.n_step_memory import NstepInputMemoryConfig
from sc2rl.runners.RunnerManager import RunnerConfig, RunnerManager

if __name__ == "__main__":

    map_name = "training_scenario_1"

    agent_conf = MultiStepActorCriticAgentConfig()
    network_conf = MultiStepInputNetworkConfig()
    brain_conf = MultiStepActorCriticBrainConfig()
    buffer_conf = NstepInputMemoryConfig()

    sample_spec = buffer_conf.memory_conf['spec']
    num_hist_steps = buffer_conf.memory_conf['N']

    agent = MultiStepActorCriticAgent(agent_conf,
                                      network_conf,
                                      brain_conf,
                                      buffer_conf)

    config = RunnerConfig(map_name=map_name, reward_func=great_victor_with_kill_bonus,
                          state_proc_func=process_game_state_to_dgl,
                          agent=agent,
                          n_hist_steps=num_hist_steps)

    runner_manager = RunnerManager(config, 2)

    wandb.init(project="sc2rl")
    wandb.config.update(agent_conf())
    wandb.config.update(network_conf())
    wandb.config.update(brain_conf())
    wandb.config.update(buffer_conf())
    wandb.watch(agent)

    iters = 0
    while iters < 1000000:
        iters += 1
        runner_manager.sample(10)
        runner_manager.transfer_sample()

        fit_return_dict = agent.fit()

        wrs = [runner.env.winning_ratio for runner in runner_manager.runners]
        mean_wr = np.mean(wrs)

        wandb.log(fit_return_dict, step=iters)
        wandb.log(fit_return_dict, step=iters)
        wandb.log({'winning_ratio': mean_wr}, step=iters)

        if iters % 50 == 0:
            save_path = os.path.join(wandb.run.dir, "model_{}".format(iters))
            torch.save(agent.state_dict(), save_path)

    runner_manager.close()
