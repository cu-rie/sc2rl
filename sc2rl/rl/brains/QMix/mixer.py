import dgl
import torch
from sc2rl.rl.networks.RelationalGraphNetwork import RelationalGraphNetwork
from sc2rl.rl.networks.FeedForward import FeedForward
from sc2rl.config.graph_configs import NODE_ALLY
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type


class QMixer(torch.nn.Module):
    def __init__(self, gnn_conf, ff_conf):
        super(QMixer, self).__init__()

        self.hyper_w_gn = RelationalGraphNetwork(gnn_conf)
        self.hyper_w_ff = FeedForward(ff_conf)
        self.hyper_v = RelationalGraphNetwork(gnn_conf)
        self.hyper_v_ff = FeedForward(ff_conf)

    def forward(self, graph, node_feature, qs,
                ally_node_type_index=NODE_ALLY):
        assert isinstance(graph, dgl.BatchedDGLGraph)

        w_emb = self.hyper_w_gn(graph, node_feature)  # [# nodes x # node_dim]
        w = self.hyper_w_ff(graph, w_emb)  # [# nodes x # 1]
        ally_node_indices = get_filtered_node_index_by_type(graph, ally_node_type_index)
        w = w[ally_node_indices, :]  # [# allies x 1]

        graph.ndata['node_feature'] = w * qs
        q_tot = dgl.sum_nodes(graph, 'node_feature')
        _ = graph.ndata.pop('node_feature')

        v_emb = self.hyper_v_gn(graph, node_feature)  # [# nodes x # node_dim]
        v = self.hyper_v_ff(graph, v_emb)  # [# nodes x # 1]

        graph.ndata['node_feature'] = v
        v = dgl.sum_nodes(graph, 'node_feature')
        _ = graph.ndata.pop('node_feature')

        q_tot = q_tot + v
        return q_tot
