import functools
import torch
from sc2rl.nn.MultiLayerPerceptron import MultiLayerPerceptron as MLP


class DiffPoolLayer(torch.nn.Module):

    def __init__(self,
                 node_dim,
                 num_neurons=[128, 128],
                 num_groups=3,
                 pooling_op='softmax',
                 spectral_norm=False):

        super(DiffPoolLayer, self).__init__()
        assert pooling_op == 'softmax' or pooling_op == 'relu', "Supported pooling ops : ['softmax', 'relu']"
        self.pooling_op = pooling_op
        self.eps = 1e-10
        self.pooler = MLP(input_dimension=node_dim, num_neurons=num_neurons, output_dimension=num_groups,
                          hidden_activation='leaky_relu',
                          out_activation='relu',
                          spectral_norm=spectral_norm)

    def forward(self, graph, node_feature):
        graph.ndata['node_feature'] = node_feature
        graph.apply_nodes(func=self.apply_node_function)
        prob = graph.ndata.pop('prob')
        return prob

    def apply_node_function(self, nodes):
        input_node_feature = nodes.data['node_feature']
        unnormalized_score = self.pooler.forward(input_node_feature)

        if self.pooling_op == 'softmax':
            normalized_score = torch.nn.functional.softmax(unnormalized_score, dim=-1)
        elif self.pooling_op == 'relu':
            numer = torch.nn.functional.relu(unnormalized_score).pow(2)
            denom = numer.sum(dim=-1, keepdim=True) + self.eps
            normalized_score = numer / denom
        else:
            raise RuntimeError("Not supported pooling mode : {}".format(self.pooling_op))

        return {'prob': normalized_score}
