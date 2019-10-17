import itertools

from sc2rl.config.unit_config import type2onehot, NUM_TOTAL_TYPES
from sc2rl.config.graph_configs import NUM_NODE_TYPES, NUM_EDGE_TYPES


def cartesian_product(*iterables, return_1d=False):
    if return_1d:
        xs = []
        ys = []
        for ij in itertools.product(*iterables):
            # if ij[0] != ij[1]:
            xs.append(ij[0])
            ys.append(ij[1])
        ret = (xs, ys)

    else:
        ret = [i for i in itertools.product(*iterables)]
    return ret


def get_one_hot_unit_type(unit_type: int):
    ret = [0] * NUM_TOTAL_TYPES
    ret[type2onehot[unit_type]] = 1.0
    return ret


def get_one_hot_node_type(node_type: int):
    ret = [0] * NUM_NODE_TYPES
    ret[node_type] = 1.0
    return ret


def get_one_hot_edge_type(edge_type: int):
    ret = [0] * NUM_EDGE_TYPES
    ret[edge_type] = 1.0
    return ret


def edge_total_damage(unit_src, unit_target):
    # WARNING: not yet tested

    # Junyoung Edited
    # Assume every unit_src has only 1 weapon
    # WARNING2 : does not cover melee attack units!

    damage = unit_src._weapons[0].damage

    if unit_src.bonus_damage is not None:
        bonus, attribute = unit_src.bonus_damage
        check = getattr(unit_target, f"is_{attributes[attribute]}")

        if check:
            damage += bonus

    return damage
