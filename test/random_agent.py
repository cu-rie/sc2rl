import wandb

from sc2rl.utils.reward_funcs import great_victor
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl
from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment

if __name__ == "__main__":
    wandb.init(project="sc2rl", name="random")

    map_name = "training_scenario_1"
    env = MicroTestEnvironment(map_name=map_name,
                               reward_func=great_victor,
                               state_proc_func=process_game_state_to_dgl)

    max_steps = 100000000
    done_counter = 0
    for i in range(max_steps):
        state, _, done = env.step(action=None)

        if done:
            done_counter += 1
            if done_counter % 20 == 0:
                wandb.log({'winning_ratio': env.winning_ratio}, step=done_counter)
