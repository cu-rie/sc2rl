from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment
from sc2rl.utils.state_process_funcs import process_game_state_to_dgl

from sc2rl.rl.modules.ActorCritic import ActorCriticModule
from sc2rl.rl.modules.Actor import ActorModule
from sc2rl.utils.graph_utils import get_largest_number_of_enemy_nodes
from sc2rl.utils.sc2_utils import nn_action_to_sc2_action


def reward_func(s, ns):
    return 1


if __name__ == "__main__":
    map_name = "3m_vs_4m_randoffset"
    test_reward_func = reward_func
    test_sate_proc_func = process_game_state_to_dgl

    env = MicroTestEnvironment(map_name, test_reward_func, test_sate_proc_func)
    ac_module = ActorCriticModule(node_input_dim=26)
    # ac_module = ActorCriticModule(node_input_dim=26)

    done_cnt = 0
    i = 0

    while True:
        # print("=========={} th iter ============".format(i))
        cur_state = env.observe()
        # for tag, unit in cur_state['tag2unit_dict'].items():
        #     print("TAG: {} | NAME: {} | Health: {} | POS: {} ".format(tag, unit.name, unit.health, unit.position))

        g = cur_state['g']
        node_feature = g.ndata.pop('node_feature')

        num_enemy = get_largest_number_of_enemy_nodes([g])
        nn_actions, info_dict = ac_module.get_action(g, node_feature, num_enemy)

        tag2unit_dict = cur_state['tag2unit_dict']
        ally_tags = info_dict['ally_tags']
        enemy_tags = info_dict['enemy_tags']

        sc2_actions = nn_action_to_sc2_action(nn_actions=nn_actions,
                                              ally_tags=ally_tags,
                                              enemy_tags=enemy_tags,
                                              tag2unit_dict=tag2unit_dict)

        loss = ac_module(g, node_feature, num_enemy)

        next_state, reward, done = env.step(action=sc2_actions)

        if done:
            done_cnt += 1
            if done_cnt >= 10:
                break

        i += 1
