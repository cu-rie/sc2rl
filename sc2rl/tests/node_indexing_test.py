import dgl
import torch
import itertools

from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment
from sc2rl.utils.state_to_graph_utils import state_proc_func
from sc2rl.rl.rl_networks.Actor import Actor


def reward_func(s, ns):
    return 1


if __name__ == "__main__":
    map_name = "training_scenario_1"
    test_reward_func = reward_func
    test_sate_proc_func = state_proc_func
    done_cnt = 0

    env = MicroTestEnvironment(map_name, test_reward_func, test_sate_proc_func)
    actor = Actor(1, 10, 1)
    while True:
        cur_state, meta_data = env.observe()
        action = Actor(cur_state)
        next_state, reward, done = env.step(action=None)
        if done:
            done_cnt += 1
            if done_cnt >= 10:
                break

    env.close()

    print("We are in the end game.")