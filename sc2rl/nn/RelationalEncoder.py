import torch

from sc2rl.nn.RelationalAttention import RelationalAttentionLayer
from sc2rl.nn.AddNorm import AddNormLayer
from sc2rl.nn.FeedForward import FeedForwardNeighbor


class RelationalEncoder(torch.nn.Module):
    def __init__(self,
                 model_dim: int,
                 use_hypernet=True,
                 hypernet_input_dim=None,
                 num_relations=None,
                 num_head: int = 3,
                 use_norm=True,
                 neighbor_degree=1,
                 num_neurons=[128, 128],
                 pooling_op='relu'):
        super(RelationalEncoder, self).__init__()

        self.attention = RelationalAttentionLayer(model_dim=model_dim,
                                                  use_hypernet=use_hypernet,
                                                  hypernet_input_dim=hypernet_input_dim,
                                                  num_relations=num_relations,
                                                  num_head=num_head,
                                                  pooling_op=pooling_op)

        self.addNorm = AddNormLayer(model_dim=model_dim,
                                    use_norm=use_norm)
        self.feedforward = FeedForwardNeighbor(model_dim=model_dim,
                                               neighbor_degree=neighbor_degree,
                                               num_neurons=num_neurons)
        self.addNorm2 = AddNormLayer(model_dim=model_dim,
                                     use_norm=use_norm)

    def forward(self, graph, node_feature, update_node_type_indices, update_edge_type_indices):
        after_attn_feature = self.attention(graph=graph,
                                            node_feature=node_feature,
                                            update_node_type_indices=update_node_type_indices,
                                            update_edge_type_indices=update_edge_type_indices)
        after_norm_feature = self.addNorm(x=node_feature,
                                          x_updated=after_attn_feature)
        after_ff_node_feature = self.feedforward(graph=graph,
                                                 node_feature=after_norm_feature)
        after_norm2_node_feature = self.addNorm2(x=after_norm_feature,
                                                 x_updated=after_ff_node_feature)

        return after_norm2_node_feature
