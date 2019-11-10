import dgl
import torch
from sc2rl.nn.MultiLayerPerceptron import MultiLayerPerceptron as MLP
from sc2rl.rl.networks.RelationalGraphNetwork import RelationalGraphNetwork
from sc2rl.rl.networks.FeedForward import FeedForward
from sc2rl.config.graph_configs import NODE_ALLY
from sc2rl.utils.graph_utils import (get_filtered_node_index_by_type,
                                     get_filtered_node_index_by_assignment)
from sc2rl.config.ConfigBase import ConfigBase


class SupQmixerConf(ConfigBase):
    def __init__(self, nn_conf=None):
        super(SupQmixerConf, self).__init__(nn_conf=nn_conf)
        self.nn_conf = {"input_dimension": 17,
                        "output_dimension": 1,
                        'num_neurons': [64, 64],
                        'spectral_norm': False
                        }


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


class SubQmixer(torch.nn.Module):
    def __init__(self, gnn_conf, ff_conf, target_assignment):
        super(SubQmixer, self).__init__()
        self.w_gn = RelationalGraphNetwork(**gnn_conf.gnn_conf)
        self.w_ff = FeedForward(ff_conf)
        self.v_gn = RelationalGraphNetwork(**gnn_conf.gnn_conf)
        self.v_ff = FeedForward(ff_conf)
        self.target_assignment = target_assignment

    def forward(self, graph, node_feature, qs,
                ally_node_type_index=NODE_ALLY):
        assert isinstance(graph, dgl.BatchedDGLGraph)

        w_emb = self.w_gn(graph, node_feature)  # [# nodes x # node_dim]
        w = torch.abs(self.w_ff(graph, w_emb))  # [# nodes x # 1]

        # Curee's trick
        ally_indices = get_filtered_node_index_by_type(graph, NODE_ALLY)
        allies_assignment = graph.ndata['assignment'][ally_indices]
        target_allies = allies_assignment == self.target_assignment
        target_indices = torch.arange(target_allies.size(0))[target_allies]

        device = w_emb.device

        _qs = torch.zeros(size=(graph.number_of_nodes(), 1), device=device)
        w = w[target_indices, :]  # [# assignments x 1]
        _qs[target_indices, :] = w * qs[target_indices].view(-1, 1)
        graph.ndata['node_feature'] = _qs
        q_tot = dgl.sum_nodes(graph, 'node_feature')
        _ = graph.ndata.pop('node_feature')

        v_emb = self.v_gn(graph, node_feature)  # [# nodes x # node_dim]
        v = self.v_ff(graph, v_emb)  # [# nodes x # 1]
        v = v[target_indices, :]  # [# allies x 1]
        _v = torch.zeros(size=(graph.number_of_nodes(), 1), device=device)
        _v[target_indices, :] = v

        graph.ndata['node_feature'] = _v
        v = dgl.sum_nodes(graph, 'node_feature')
        _ = graph.ndata.pop('node_feature')

        q_tot = q_tot + v
        return q_tot.view(-1)

class Soft_SubQmixer(torch.nn.Module):
    def __init__(self, gnn_conf, ff_conf, target_assignment):
        super(Soft_SubQmixer, self).__init__()
        self.w_gn = RelationalGraphNetwork(**gnn_conf.gnn_conf)
        self.w_ff = FeedForward(ff_conf)
        self.v_gn = RelationalGraphNetwork(**gnn_conf.gnn_conf)
        self.v_ff = FeedForward(ff_conf)
        self.target_assignment = target_assignment

    def forward(self, graph, node_feature, qs,
                ally_node_type_index=NODE_ALLY):
        assert isinstance(graph, dgl.BatchedDGLGraph)

        w_emb = self.w_gn(graph, node_feature)  # [# nodes x # node_dim]
        w = torch.abs(self.w_ff(graph, w_emb))  # [# nodes x # 1]

        # Curee's trick
        ally_indices = get_filtered_node_index_by_type(graph, NODE_ALLY)
        target_assignment_weight = graph.ndata['normalized_score'][ally_indices]
        # target_allies = allies_assignment == self.target_assignment
        # target_indices = torch.arange(target_allies.size(0))[target_allies]

        device = w_emb.device

        _qs = torch.zeros(size=(graph.number_of_nodes(), 1), device=device)
        w = w[ally_indices, :]  # [# assignments x 1]
        _qs[ally_indices, :] = w * qs[ally_indices].view(-1, 1)
        graph.ndata['node_feature'] = _qs
        q_tot = dgl.sum_nodes(graph, 'node_feature')
        _ = graph.ndata.pop('node_feature')

        v_emb = self.v_gn(graph, node_feature)  # [# nodes x # node_dim]
        v = self.v_ff(graph, v_emb)  # [# nodes x # 1]
        v = v[ally_indices, :]  # [# allies x 1]
        _v = torch.zeros(size=(graph.number_of_nodes(), 1), device=device)
        _v[ally_indices, :] = v

        graph.ndata['node_feature'] = _v
        v = dgl.sum_nodes(graph, 'node_feature')
        _ = graph.ndata.pop('node_feature')

        q_tot = q_tot + v
        return q_tot.view(-1)


class SupQmixer(torch.nn.Module):
    def __init__(self, input_dim, conf):
        super(SupQmixer, self).__init__()
        nn_conf = conf.nn_conf
        nn_conf['input_dimension'] = input_dim
        self.w = MLP(**nn_conf)
        self.v = MLP(**nn_conf)

    def forward(self, graph, node_feature, sub_q_tots):
        graph.ndata['node_feature'] = node_feature
        device = node_feature.device

        q_tot = torch.zeros(graph.batch_size, device=device)
        for i, sub_q_tot in enumerate(sub_q_tots):
            node_indices = get_filtered_node_index_by_assignment(graph, i)
            mask = torch.zeros(size=(node_feature.shape[0], 1), device=device)
            mask[node_indices, :] = 1

            graph.ndata['masked_node_feature'] = graph.ndata['node_feature'] * mask
            w_input = dgl.sum_nodes(graph, 'masked_node_feature')
            q_tot = q_tot + torch.abs(self.w(w_input)).view(-1) * sub_q_tot
            _ = graph.ndata.pop('masked_node_feature')

        v = self.v(dgl.sum_nodes(graph, 'node_feature')).view(-1)
        q_tot = q_tot + v
        return q_tot


if __name__ == "__main__":
    QMixer()
