from functools import partial
import torch
from torch_scatter import scatter_add

from .Hypernet import HyperNetwork


class RelationalAttentionLayer(torch.nn.Module):

    def __init__(self,
                 model_dim: int,
                 use_hypernet=True,
                 hypernet_input_dim=None,
                 num_relations=None,
                 concat_self_o=False,
                 num_head: int = 2,
                 pooling_op='relu'):
        super(RelationalAttentionLayer, self).__init__()

        if use_hypernet:
            assert num_relations is None, "when 'use_hypernet' = True, num_relations will be ignored."
            assert hypernet_input_dim is not None, "when 'use_hypernet' = True, hypernet_input_dim should be specified."

        if not use_hypernet:
            assert type(num_relations) == int, "when 'use_hypernet' = False, num_relations should be integer."

        self.model_dim = model_dim
        self.num_head = num_head
        self.use_hypernet = use_hypernet
        self.num_relations = num_relations
        assert pooling_op == 'softmax' or pooling_op == 'relu', "Supported pooling ops : ['softmax', 'relu']"
        self.pooling_op = pooling_op
        self.eps = 1e-10
        self.concat_self_o = concat_self_o

        if self.concat_self_o:
            o_input_dim = model_dim * (num_head + 1)
        else:
            o_input_dim = model_dim * num_head

        if self.use_hypernet:
            # Initialize hyper-networks for constructing query, key, value matrix dynamically.

            self.WQ = torch.nn.Linear(model_dim, num_head * model_dim, bias=False)
            self.WK = HyperNetwork(num_rows=model_dim * num_head,
                                   num_cols=model_dim,
                                   input_dimension=hypernet_input_dim,
                                   output_dimension=model_dim * model_dim * num_head)
            self.WV = HyperNetwork(num_rows=model_dim * num_head,
                                   num_cols=model_dim,
                                   input_dimension=hypernet_input_dim,
                                   output_dimension=model_dim * model_dim * num_head)

            self.WO = torch.nn.Linear(o_input_dim, model_dim, bias=False)

        else:
            wk_dict = {}
            wv_dict = {}
            for i in range(self.num_relations):
                wk_dict['WK{}'.format(i)] = torch.nn.Linear(model_dim, num_head * model_dim, bias=False)
                wv_dict['WV{}'.format(i)] = torch.nn.Linear(model_dim, num_head * model_dim, bias=False)

            self.WQ = torch.nn.Linear(model_dim, num_head * model_dim, bias=False)
            self.WK = torch.nn.ModuleDict(wk_dict)
            self.WV = torch.nn.ModuleDict(wv_dict)
            self.WO = torch.nn.Linear(o_input_dim, model_dim, bias=False)

    def forward(self, graph, feature_dict, update_node_types=['ally'], skip_edge_types=['attack_edge']):
        """
        :param graph: Structure only graph. Input graph has no node features
        :param node_feature: Tensor. Node features
        :param device: str. device flag
        :return: updated features
        """

        for key, val in feature_dict.items():
            graph.nodes[key].data['node_feature'] = val

        executable_edge_types = graph.etypes
        for remove_edge_type in skip_edge_types:
            executable_edge_types.remove(remove_edge_type)

        for i, etype in enumerate(executable_edge_types):
            message_func = partial(self.message_function, etype_idx=i)
            reduce_func = partial(self.reduce_function, etype_idx=i)
            graph.send_and_recv(graph[etype].edges(), message_func=message_func, reduce_func=reduce_func, etype=etype)

        apply_func = partial(self.apply_node_function, num_etypes=len(executable_edge_types))

        for ntype in update_node_types:
            graph.apply_nodes(apply_func, ntype=ntype)

        ret_dict = dict()
        for ntype in graph.ntypes:
            ret_dict[ntype] = graph.nodes[ntype].data.pop('node_feature')

        return ret_dict

    def message_function(self, edges, etype_idx):
        src_node_features = edges.src['node_feature']  # [Num. Edges x Model_dim]

        if self.use_hypernet:
            src_node_features = src_node_features.unsqueeze(-1)  # [#.Edges x Model_dim x 1]
            WK = self.WK(etype_idx)
            WV = self.WV(etype_idx)

            keys = torch.bmm(WK, src_node_features)  # [#.Edges x (Model_dim x #.head) x 1]
            keys = keys.squeeze()  # [#.Edges x (Model_dim x #.head)]
            values = torch.bmm(WV, src_node_features)  # [#.Edges x (Model_dim x #.head) x 1]
            values = values.squeeze()  # [#.Edges x (Model_dim x #.head)]
        else:
            WK = self.WK['WK{}'.format(etype_idx)]
            WV = self.WV['WV{}'.format(etype_idx)]
            keys = WK(src_node_features)
            values = WV(src_node_features)

        return {'key{}'.format(etype_idx): keys, 'value{}'.format(etype_idx): values}

    def reduce_function(self, nodes, etype_idx):
        key = nodes.mailbox['key{}'.format(etype_idx)]  # [(Batched) Node x (Model_dim x #.head)]
        value = nodes.mailbox['value{}'.format(etype_idx)]  # [(Batched) Node x (Model_dim x #.head)]
        return {'key{}'.format(etype_idx): key, 'value{}'.format(etype_idx): value}

    def apply_node_function(self, nodes, num_etypes):

        node_features = nodes.data['node_feature']
        queries = self.WQ(node_features)  # [(Batched) Node x (Model_dim x #.head)]

        keys = []
        values = []

        for i in range(num_etypes):
            key = nodes.data.pop('key{}'.format(i))
            value = nodes.data.pop('value{}'.format(i))
            keys.append(key)
            values.append(value)

        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)

        node_counter = 0
        node_indices = []
        num_edges = []
        expanded_queries = []
        for query, key in zip(queries, keys):
            _num_edges = key.shape[0]
            expanded_queries.extend([query] * _num_edges)
            node_indices.extend([node_counter] * _num_edges)
            node_counter += 1
            num_edges.append(_num_edges)

        node_indices = torch.tensor(node_indices).long()
        expanded_queries = torch.stack(expanded_queries)  # [#.Edges x (Model_dim x #.head)]
        expanded_keys = keys.view(-1, self.model_dim * self.num_head)  # [#.Edges x (Model_dim x #.head)]

        scores = expanded_queries * expanded_keys  # [#.Edges x (Model_dim x #.head)]

        scores = scores.view(-1, self.num_head, self.model_dim)  # [#.Edges x Model_dim x #.head]
        scores = scores.sum(2)  # [#.Edges x #.head]
        scores = scores / torch.sqrt(torch.Tensor((self.model_dim * self.num_head,)))  # [#.Edges x #.head]

        # Compute Score
        if self.pooling_op == 'softmax':
            numer = torch.exp(scores)
            denom = scatter_add(numer, node_indices, dim=0)
        elif self.pooling_op == 'relu':
            numer = torch.nn.functional.relu(scores).pow(2) + self.eps
            denom = scatter_add(numer, node_indices, dim=0)
        else:
            raise RuntimeError("No way! '{}' cannot be true".format(self.pooling_op))

        denom_repeat = torch.repeat_interleave(denom, torch.tensor(num_edges), dim=0)
        normalized_score = numer / denom_repeat
        normalized_score = normalized_score.view(-1, self.num_head, 1)
        values_scatter = values.view(-1, self.num_head, self.model_dim)
        weighted_values = normalized_score * values_scatter
        z = scatter_add(weighted_values, node_indices, dim=0)

        z = z.reshape(-1, self.num_head * self.model_dim)

        if self.concat_self_o:
            z = torch.cat([z, nodes.data.pop('node_feature')], dim=1)

        out = self.WO(z)

        return {'node_feature': out}
