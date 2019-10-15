from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment
from sc2rl.utils.state_to_graph import process_game_state_to_dgl
from sc2rl.rl.rl_modules.ActionModules import AttackModule
from sc2rl.config.graph_configs import EDGE_IN_ATTACK_RANGE, EDGE_ENEMY, NODE_ENEMY


def reward_func(s, ns):
    return 1


if __name__ == "__main__":
    map_name = "3m_vs_4m_randoffset"
    test_reward_func = reward_func
    test_sate_proc_func = process_game_state_to_dgl

    env = MicroTestEnvironment(map_name, test_reward_func, test_sate_proc_func)
    attack_module = AttackModule(node_dim=26)

    done_cnt = 0
    i = 0

    while True:
        print("=========={} th iter ============".format(i))
        cur_state = env.observe()
        for tag, unit in cur_state['tag2unit_dict'].items():
            print("TAG: {} | NAME: {} | Health: {} | POS: {} ".format(tag, unit.name, unit.health, unit.position))

        g = cur_state['g']
        g_feature = g.ndata.pop('node_feature')
        attack_argument = attack_module.get_action(g, g_feature, EDGE_IN_ATTACK_RANGE, NODE_ENEMY)
        next_state, reward, done = env.step(action=None)

        if done:
            done_cnt += 1
            if done_cnt >= 10:
                break

        i += 1