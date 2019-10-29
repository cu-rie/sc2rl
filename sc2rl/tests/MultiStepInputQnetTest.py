from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment

from sc2rl.utils.reward_funcs import great_victor_with_kill_bonus
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl
from sc2rl.utils.HistoryManagers import HistoryManager

from sc2rl.memory.n_step_memory import NstepInputMemoryConfig

if __name__ == "__main__":
    map_name = "training_scenario_1"
    env = MicroTestEnvironment(map_name=map_name,
                               reward_func=great_victor_with_kill_bonus,
                               state_proc_func=process_game_state_to_dgl)

    buffer_conf = NstepInputMemoryConfig()

    init_graph = env.observe()['g']
    history_manager = HistoryManager(
        n_hist_steps=buffer_conf.memory_conf['N'], init_graph=init_graph)

