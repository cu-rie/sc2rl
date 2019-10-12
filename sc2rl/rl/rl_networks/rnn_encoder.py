import dgl
import torch


class RNNEncoder(torch.nn.Module):

    def __init__(self, rnn, one_step_encoder):
        super(RNNEncoder, self).__init__()
        self.rnn = rnn
        self.one_step_encoder = one_step_encoder

    def forward(self,
                batch_size: int,
                num_time_steps: int,
                batch_time_batched_graph: dgl.BatchedDGLGraph,
                feature_dict: dict):

        embedded_feature_dict = self.one_step_encoder(batch_time_batched_graph, feature_dict)

        for key, val in embedded_feature_dict.items():
            batch_time_batched_graph.nodes[key].data['node_feature'] = val

        readouts = dgl.sum_nodes(batch_time_batched_graph, 'node_feature')
        readouts = readouts.view(batch_size, num_time_steps, 1)
        global_feature = self.rnn(readouts)

        return global_feature
