import torch
from sc2rl.nn.RelationalEncoder import RelationalEncoder


class RelationalNetwork(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 model_dim,
                 use_hypernet,
                 hypernet_input_dim,
                 num_relations,
                 num_head,
                 use_norm,
                 neighbor_degree,
                 num_neurons,
                 pooling_op):
        super(RelationalNetwork, self).__init__()

        layers = []
        for i in range(num_layers):
            layer = RelationalEncoder(model_dim=model_dim,
                                      use_hypernet=use_hypernet,
                                      hypernet_input_dim=hypernet_input_dim,
                                      num_relations=num_relations,
                                      num_head=num_head,
                                      use_norm=use_norm,
                                      neighbor_degree=neighbor_degree,
                                      num_neurons=num_neurons,
                                      pooling_op=pooling_op)
            layers.append(layer)

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, graph):

        feature_dict = dict()
        for ntype in graph.ntypes:
            feature_dict[ntype] = graph.nodes[ntype].data.pop('node_feature')

        for layer in self.layers:
            node_feature_dict = layer(graph, feature_dict)

        return node_feature_dict
