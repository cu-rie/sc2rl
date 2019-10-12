import torch

from sc2rl.nn.RelationalAttention import RelationalAttentionLayer
from sc2rl.nn.AddNorm import AddNormLayerHetero
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

        self.addNorm = AddNormLayerHetero(model_dim=model_dim,
                                          use_norm=use_norm)
        self.feedforward = FeedForwardNeighbor(model_dim=model_dim,
                                               neighbor_degree=neighbor_degree,
                                               num_neurons=num_neurons)
        self.addNorm2 = AddNormLayerHetero(model_dim=model_dim,
                                           use_norm=use_norm)

    def forward(self, graph, feature_dict):
        after_attn_feature_dict = self.attention.forward(graph=graph,
                                                         feature_dict=feature_dict)
        after_norm_feature_dict = self.addNorm.forward(x_dict=feature_dict,
                                                       x_updated_dict=after_attn_feature_dict)
        after_ff_node_feat = self.feedforward.forward(graph=graph,
                                                      feature_dict=after_norm_feature_dict)
        after_norm2_node_feat = self.addNorm2.forward(x_dict=after_norm_node_feat,
                                                      x_updated_dict=after_ff_node_feat)

        return after_norm2_node_feat
