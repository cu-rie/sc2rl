# Basic reward function structure
#
# def some_reward_func(state_dict, next_state_dict, done, *args, **kwargs):
#     some calculation
#     return reward
#
# *args, **kwargs will be fed in runtime with functool.partial


def great_victor_with_kill_bonus(state_dict,
                                 next_state_dict,
                                 done,
                                 victory_coeff=100.0,
                                 ally_killed_penalty_coeff=5.0,
                                 enemy_kill_bonus_coeff=5.0):
    units = state_dict['units']
    next_units = next_state_dict['units']

    num_allies = len(units.owned)
    num_allies_after = len(next_units.owned)

    num_enemies = len(units.enemy)
    num_enemies_after = len(next_units.enemy)

    enemies_killed = num_enemies - num_enemies_after
    allies_killed = num_allies - num_allies_after

    reward = enemy_kill_bonus_coeff * enemies_killed - ally_killed_penalty_coeff * allies_killed

    if done:
        reward = victory_coeff * (num_allies_after - num_enemies_after)

    return reward


def victory(state_dict,
            next_state_dict,
            done,
            victory_coeff=1.0):
    next_units = next_state_dict['units']

    num_allies_after = len(next_units.owned)
    num_enemies_after = len(next_units.enemy)

    win = num_allies_after > num_enemies_after

    if done:
        if win:
            reward = victory_coeff
        else:
            reward = -victory_coeff
    else:
        reward = 0

    return reward


def victory_if_zero_enemy(stat_dict,
                          next_state_dict,
                          done):
    next_units = next_state_dict['units']
    num_enemies_after = len(next_units.enemy)

    win = num_enemies_after != 0

    if done:
        if win:
            reward = 1.0
        else:
            reward = -1.0
    else:
        reward = 0

    return reward


def great_victor(state_dict,
                 next_state_dict,
                 done,
                 victory_coeff=1.0):
    next_units = next_state_dict['units']
    num_allies_after = len(next_units.owned)
    num_enemies_after = len(next_units.enemy)

    if done:
        reward = victory_coeff * (num_allies_after - num_enemies_after)
    else:
        reward = 0.0

    return reward


def kill_bonus_reward(state_dict,
                      next_state_dict,
                      done,  # for interfacing
                      ally_reserve_bonus_coeff=5.0,
                      enemy_kill_bonus_coeff=10.0):
    units = state_dict['units']
    next_units = next_state_dict['units']

    num_allies = len(units.owned)
    num_allies_after = len(next_units.owned)

    num_enemies = len(units.enemy)
    num_enemies_after = len(next_units.enemy)

    enemies_killed = num_enemies - num_enemies_after
    allies_killed = num_allies - num_allies_after

    reward = enemy_kill_bonus_coeff * enemies_killed + ally_reserve_bonus_coeff * allies_killed
    return reward
