import dgl
import torch
import itertools

from sc2rl.environments.MicroTestEnvironment import MicroTestEnvironment


def reward_func(s, ns):
    return 1


def state_proc_func(game_state):
    g = dgl.DGLGraph()
    units = game_state.units
    tags = [u.tag for u in units]
    num_units = len(units)

    g.add_nodes(num_units, {'tag': torch.tensor(tags),
                            'x': torch.arange(0, num_units)})
    u, v = cartesian_product(range(num_units), range(num_units), return_1d=True)
    g.add_edges(u, v)
    return g


def cartesian_product(*iterables, return_1d=False):
    if return_1d:
        xs = []
        ys = []
        for ij in itertools.product(*iterables):
            if ij[0] != ij[1]:
                xs.append(ij[0])
                ys.append(ij[1])
        ret = (xs, ys)

    else:
        ret = [i for i in itertools.product(*iterables) if i[0] != i[1]]
    return ret


def pull_graph(graph):
    def meassage_func(edges):
        x = edges.src['x']
        return {'x': x}

    def reduce_func(nodes):
        updated_x = nodes.mailbox['x']
        return {'x': updated_x}

    graph.pull(graph.nodes(),
               message_func=meassage_func,
               reduce_func=reduce_func)

    return graph


if __name__ == "__main__":
    map_name = "training_scenario_1"
    test_reward_func = reward_func
    test_sate_proc_func = state_proc_func
    done_cnt = 0

    env = MicroTestEnvironment(map_name, test_reward_func, test_sate_proc_func)
    while True:
        cur_state = env.observe()
        graph = pull_graph(cur_state)
        next_state, reward, done = env.step(action=None)
        if done:
            done_cnt += 1
            if done_cnt >= 10:
                break

    env.close()

    print("We are in the end game.")
