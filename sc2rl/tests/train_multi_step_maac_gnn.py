import wandb
import numpy as np

from sc2rl.utils.reward_funcs import great_victor_with_kill_bonus
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl

from sc2rl.rl.agents.MultiStepActorCriticAgent import MultiStepActorCriticAgent, MultiStepActorCriticAgentConfig
from sc2rl.rl.brains.MultiStepInputActorCriticBrain_refac import MultiStepActorCriticBrainConfig
from sc2rl.rl.networks.MultiStepInputGraphNetwork import MultiStepInputGraphNetworkConfig

from sc2rl.memory.n_step_memory import NstepInputMemoryConfig
from sc2rl.runners.RunnerManager import RunnerConfig, RunnerManager

if __name__ == "__main__":

    map_name = "training_scenario_1"

    agent_conf = MultiStepActorCriticAgentConfig()
    network_conf = MultiStepInputGraphNetworkConfig()
    brain_conf = MultiStepActorCriticBrainConfig()
    buffer_conf = NstepInputMemoryConfig()
    use_attention = False
    num_runners = 5
    num_samples = 10

    sample_spec = buffer_conf.memory_conf['spec']
    num_hist_steps = buffer_conf.memory_conf['N']

    agent = MultiStepActorCriticAgent(agent_conf,
                                      network_conf,
                                      brain_conf,
                                      buffer_conf,
                                      use_attention=use_attention)

    config = RunnerConfig(map_name=map_name, reward_func=great_victor_with_kill_bonus,
                          state_proc_func=process_game_state_to_dgl,
                          agent=agent,
                          n_hist_steps=num_hist_steps)

    runner_manager = RunnerManager(config, num_runners)

    wandb.init(project="sc2rl")
    wandb.config.update({'use_attention': use_attention,
                         'num_runners': num_runners,
                         'num_samples': num_samples})
    wandb.config.update(agent_conf())
    wandb.config.update(network_conf())
    wandb.config.update(brain_conf())
    wandb.config.update(buffer_conf())

    iters = 0
    while iters < 1000000:
        iters += 1
        runner_manager.sample(num_samples)
        runner_manager.transfer_sample()
        print("fit at {}".format(iters))
        fit_return_dict = agent.fit()
        wandb.log(fit_return_dict, step=iters)
        wrs = [runner.env.winning_ratio for runner in runner_manager.runners]
        mean_wr = np.mean(wrs)

        wandb.log(fit_return_dict, step=iters)
        wandb.log({'winning_ratio': mean_wr}, step=iters)

    runner_manager.close()
