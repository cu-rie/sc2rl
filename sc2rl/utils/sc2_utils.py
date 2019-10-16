import math
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type
from sc2rl.config.graph_configs import NODE_ENEMY


def get_move_position(unit_position, move_dir, cardinal_points=4, radius=10):
    # when cardinal_points = 4:
    # move_dir = 0 -> RIGHT
    # move_dir = 1 -> UP
    # move_dir = 2 -> LEFT
    # move_dit = 3 -> DOWN

    theta = 2 * math.pi * float(move_dir) / cardinal_points
    delta = (math.cos(theta) * radius, math.sin(theta) * radius)
    position = unit_position + delta
    return position


def nn_action_to_sc2_action(nn_actions, ally_tags, enemy_tags, tag2unit_dict, move_dim=4):
    sc_action_list = list()

    for nn_action, ally_tag, enemy_tag in zip(nn_actions, ally_tags, enemy_tags):
        unit = tag2unit_dict[int(ally_tag)]
        if nn_action <= move_dim - 1:
            move_point = get_move_position(unit.position, nn_action)
            action = unit.move(move_point)
            print("[MOVE] Unit Tag:{}".format(ally_tag))
            print("[MOVE] MOVE DIR {}".format(nn_action))
            print("[MOVE] Unit POS before move:{}".format(unit.position))
            print("[MOVE] Expected POS after move {}".format(move_point))
        elif nn_action == move_dim:
            action = unit.hold_position()
            print("[HOLD] Unit Tag:{}".format(ally_tag))
            print("[HOLD] Unit POS before hold:{}".format(unit.position))
        else: #if move_dim + 1 <= nn_action <= move_dim + num_enemies:
            enemy_tag = enemy_tag[nn_action - move_dim]
            enemy_unit = tag2unit_dict[int(enemy_tag)]
            action = unit.attack(enemy_unit)
            print("[ATTACK] Unit Tag:{}".format(ally_tag))
            print("[ATTACK] Unit Attack WHOM:{}".format(enemy_tag))
            print("[ATTACK] Unit Attack WHOM HP Before:{}".format(enemy_unit.health))

        sc_action_list.append(action)
    return sc_action_list
