import dgl
import torch
from sc2rl.rl.networks.RelationalGraphNetwork import RelationalGraphNetwork
from sc2rl.config.graph_configs import NODE_ALLY
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type


class QMixer(torch.nn.Module):

    def __init__(self,
                 conf):
        super(QMixer, self).__init__()

        self.embedding_dim = conf.mixer_conf['embedding_dim']

        self.hyper_w = RelationalGraphNetwork(conf.hypernet_conf)
        self.hyper_v = RelationalGraphNetwork(conf.hypernet_conf)

    def forward(self, graph, node_feature, qs,
                ally_node_type_index=NODE_ALLY):
        w = self.hyper_w(graph, node_feature)  # [# nodes x 1]
        ally_node_indices = get_filtered_node_index_by_type(graph, ally_node_type_index)
        w = w[ally_node_indices, :]  # [# allies x 1]

        v = self.hyper_w(graph, node_feature)  # [# graphs x 1]
        graph.ndata['node_feature'] = w * qs

        assert isinstance(graph, dgl.BatchedDGLGraph)
        q_tot = dgl.sum_nodes(graph, 'node_feature')
        q_tot = q_tot + v
        return q_tot
