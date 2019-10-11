import dgl
import numpy as np
import torch
import itertools

from sc2rl.config.unit_config import type2cost, type2onehot, NUM_TOTAL_TYPES
from sc2rl.config.graph_configs import NUM_NODE_TYPES, NUM_EDGE_TYPES, NODE_ALLIES, NODE_ENEMY
from sc2.game_state import GameState


def cartesian_product_(*iterables, return_1d=False):
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


def state_proc_func(state):
    ally_units = state.units.owned
    enemy_units = state.units.enemy

    num_allies = len(ally_units)
    num_enemies = len(enemy_units)

    tag2unit_dict = dict()

    ally_node_features = []

    allies_mineral_cost = 0
    allies_vespene_cost = 0
    allies_food_cost = 0

    exist_allies = num_allies >= 1
    exist_enemies = num_enemies >= 1

    if exist_allies:
        allies_center_pos = ally_units.center
        allies_tag = []
        for i, allies_unit in enumerate(ally_units):
            tag2unit_dict[allies_unit.tag] = allies_unit
            node_feature = list()
            one_hot_type_id = get_one_hot_unit_type(allies_unit.type_id.value)
            node_feature.extend(one_hot_type_id)
            node_feature.extend(list(allies_unit.position))
            node_feature.extend(list(allies_center_pos - allies_unit.position))
            node_feature.append(allies_unit.health)
            node_feature.append(allies_unit.health_max)
            node_feature.append(allies_unit.health_percentage)
            node_feature.append(allies_unit.weapon_cooldown)
            node_feature.append(allies_unit.ground_dps)
            one_hot_node_type = get_one_hot_node_type(NODE_ALLIES)
            node_feature.extend(one_hot_node_type)
            ally_node_features.append(node_feature)

            allies_tag.append(allies_unit.tag)

            allies_mineral_cost += type2cost[allies_unit.name][0]
            allies_vespene_cost += type2cost[allies_unit.name][1]
            allies_food_cost += type2cost[allies_unit.name][2]

    enemies_mineral_cost = 0
    enemies_vespene_cost = 0
    enemies_food_cost = 0
    enemy_node_features = []

    if exist_enemies:
        enemy_center_pos = enemy_units.center
        enemies_tag = []
        for j, enemy_unit in enumerate(enemy_units):
            tag2unit_dict[enemy_unit.tag] = enemy_unit
            node_feature = list()
            one_hot_type_id = get_one_hot_unit_type(enemy_unit.type_id.value)
            node_feature.extend(one_hot_type_id)
            node_feature.extend(list(enemy_unit.position))
            node_feature.extend(list(enemy_center_pos - enemy_unit.position))
            node_feature.append(enemy_unit.health)
            node_feature.append(enemy_unit.health_max)
            node_feature.append(enemy_unit.health_percentage)
            node_feature.append(enemy_unit.weapon_cooldown)
            node_feature.append(enemy_unit.ground_dps)
            one_hot_node_type = get_one_hot_node_type(NODE_ENEMY)
            node_feature.extend(one_hot_node_type)
            enemy_node_features.append(node_feature)

            enemies_tag.append(enemy_unit.tag)

            enemies_mineral_cost += type2cost[enemy_unit.name][0]
            enemies_vespene_cost += type2cost[enemy_unit.name][1]
            enemies_food_cost += type2cost[enemy_unit.name][2]

    ally_node_features = torch.Tensor(np.stack(ally_node_features))
    enemy_node_features = torch.Tensor(np.stack(enemy_node_features))

    allies_indices = range(num_allies)
    enemy_indices = range(num_enemies)

    if num_allies >= 2:
        allies_edge = cartesian_product_(allies_indices, allies_indices, return_1d=False)
        g_allies = dgl.heterograph({('ally', 'ally_edge', 'ally'): allies_edge})
    else:
        g_allies = dgl.DGLGraph()

    if exist_enemies and exist_enemies:
        enemy_edge = cartesian_product_(enemy_indices, allies_indices, return_1d=False)
        g_enemy = dgl.heterograph({('enemy', 'enemy_edge', 'ally'): enemy_edge})
    else:
        g_enemy = dgl.DGLGraph()

    if exist_allies and exist_enemies:
        attack_edge = cartesian_product_(enemy_indices, allies_indices, return_1d=False)
        g_attack = dgl.heterograph({('enemy', 'attack_edge', 'ally'): attack_edge})
    else:
        g_attack = dgl.DGLGraph()

    graph = dgl.hetero_from_relations([g_allies, g_enemy, g_attack])
    if exist_allies:
        graph.nodes['ally'].data['node_feature'] = ally_node_features
        graph.nodes['ally'].data['tag'] = torch.Tensor(allies_tag)
    if exist_enemies:
        graph.nodes['enemy'].data['node_feature'] = enemy_node_features
        graph.nodes['enemy'].data['tag'] = torch.Tensor(enemies_tag)

    _gf = [allies_mineral_cost,
           allies_vespene_cost,
           allies_food_cost,
           enemies_mineral_cost,
           enemies_vespene_cost,
           enemies_food_cost]
    global_feature = torch.Tensor(data=_gf).view(1, -1)

    meta_data = dict()
    meta_data['tag2unit_dict'] = tag2unit_dict
    meta_data['global_feature'] = global_feature
    return graph, meta_data
