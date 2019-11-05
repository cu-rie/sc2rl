import dgl
import torch
from sc2rl.rl.networks.RelationalGraphNetwork import RelationalGraphNetwork, RelationalGraphNetworkConfig
from sc2rl.rl.networks.FeedForward import FeedForward, FeedForwardConfig
from sc2rl.config.graph_configs import NODE_ALLY
from sc2rl.utils.graph_utils import get_filtered_node_index_by_type
from sc2rl.config.ConfigBase import ConfigBase


class QMixer(torch.nn.Module):
    def __init__(self, gnn_conf, ff_conf):
        super(QMixer, self).__init__()

        self.w_gn = RelationalGraphNetwork(**gnn_conf.gnn_conf)
        self.w_ff = FeedForward(ff_conf)
        self.v_gn = RelationalGraphNetwork(**gnn_conf.gnn_conf)
        self.v_ff = FeedForward(ff_conf)

    def forward(self, graph, node_feature, qs,
                ally_node_type_index=NODE_ALLY):
        assert isinstance(graph, dgl.BatchedDGLGraph)

        w_emb = self.w_gn(graph, node_feature)  # [# nodes x # node_dim]
        w = torch.abs(self.w_ff(graph, w_emb))  # [# nodes x # 1]
        ally_node_indices = get_filtered_node_index_by_type(graph, ally_node_type_index)

        device = w_emb.device

        _qs = torch.zeros(size=(graph.number_of_nodes(), 1), device=device)
        w = w[ally_node_indices, :]  # [# allies x 1]
        _qs[ally_node_indices, :] = w * qs.view(-1, 1)
        graph.ndata['node_feature'] = _qs
        q_tot = dgl.sum_nodes(graph, 'node_feature')
        _ = graph.ndata.pop('node_feature')

        v_emb = self.v_gn(graph, node_feature)  # [# nodes x # node_dim]
        v = self.v_ff(graph, v_emb)  # [# nodes x # 1]
        v = v[ally_node_indices, :]  # [# allies x 1]
        _v = torch.zeros(size=(graph.number_of_nodes(), 1), device=device)
        _v[ally_node_indices, :] = v

        graph.ndata['node_feature'] = _v
        v = dgl.sum_nodes(graph, 'node_feature')
        _ = graph.ndata.pop('node_feature')

        q_tot = q_tot + v
        return q_tot.view(-1)


if __name__ == "__main__":
    QMixer()
