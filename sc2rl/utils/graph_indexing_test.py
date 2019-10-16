import dgl
import torch


def get_graph(n):
    g = dgl.DGLGraph()
    g.add_nodes(n)
    g.ndata['feat'] = torch.ones(n, 1) * n
    return g


if __name__ == "__main__":

    rnn = torch.nn.GRU(input_size=1,
                       hidden_size=10,
                       num_layers=2,
                       batch_first=True)

    gs = list()
    n_graphs = 10
    for i in range(1, n_graphs+1, 1):
        gs.append(get_graph(i))

    batched_g = dgl.batch(gs)
    readouts = dgl.sum_nodes(batched_g, 'feat')
    print(readouts)

    time_steps = 5
    readouts = readouts.view(-1, 5, readouts.shape[-1])
    print(readouts)

    global_feat = rnn(readouts)

    print(global_feat)
