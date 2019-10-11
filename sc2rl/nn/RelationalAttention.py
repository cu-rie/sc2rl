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

    def forward(self, graph, node_feature, device):
        """
        :param graph: Structure only graph. Input graph has no node features
        :param node_feature: Tensor. Node features
        :param device: str. device flag
        :return: updated features
        """
        graph.ndata['node_feature'] = node_feature
        message_func = partial(self.message_function, device=device)
        reduce_func = partial(self.reduce_function, device=device)

        graph.update_all(message_func=message_func,
                         reduce_func=reduce_func,
                         apply_node_func=self.apply_node_function)

        # Delete intermediate feature to maintain structure only graph
        _ = graph.ndata.pop('node_feature')
        _ = graph.ndata.pop('z')

        return graph.ndata.pop('updated_node_feature')

    def message_function(self, edges, device):
        src_node_features = edges.src['node_feature']  # [Num. Edges x Model_dim]

        if self.use_hypernet:
            edge_types = edges.data['edge_type_one_hot']
            src_node_features = src_node_features.unsqueeze(-1)  # [#.Edges x Model_dim x 1]
            edge_types = edge_types.float()  # [#.Egdes x 1]
            WK = self.WK(edge_types)  # [#.Edges x (Model_dim x #.head) x Model_dim]
            WV = self.WV(edge_types)  # [#.Edges x (Model_dim x #.head) x Model_dim]

            keys = torch.bmm(WK, src_node_features)  # [#.Edges x (Model_dim x #.head) x 1]
            keys = keys.squeeze()  # [#.Edges x (Model_dim x #.head)]
            values = torch.bmm(WV, src_node_features)  # [#.Edges x (Model_dim x #.head) x 1]
            values = values.squeeze()  # [#.Edges x (Model_dim x #.head)]
        else:
            edge_types = edges.data['edge_type']
            keys = torch.zeros(self.model_dim * self.num_head,
                               src_node_features.shape[0])  # [(Model_dim x #.head) x #.Edges]

            values = torch.zeros(self.model_dim * self.num_head,
                                 src_node_features.shape[0])  # [(Model_dim x #.head) x #.Edges]
            for i in range(self.num_relations):
                WK = self.WK['WK{}'.format(i)]
                WV = self.WV['WV{}'.format(i)]
                curr_relation_mask = edge_types == i
                curr_relation_post = torch.arange(src_node_features.shape[0])[curr_relation_mask]
                if curr_relation_mask.sum() == 0:
                    continue
                cur_node_features = src_node_features[curr_relation_mask]
                keys[:, curr_relation_post] = WK(cur_node_features).t()
                values[:, curr_relation_post] = WV(cur_node_features).t()
            keys = keys.t()  # [#.Edges x (Model_dim x #.head)]
            values = values.t()  # [#.Edges x (Model_dim x #.head)]

        return {'key': keys, 'value': values}

    def reduce_function(self, nodes, device):
        node_features = nodes.data['node_feature']
        queries = self.WQ(node_features)  # [(Batched) Node x (Model_dim x #.head)]

        keys = nodes.mailbox['key']  # [(Batched) Node x (Model_dim x #.head)]
        values = nodes.mailbox['value']  # [(Batched) Node x (Model_dim x #.head)]

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
        ret_z = scatter_add(weighted_values, node_indices, dim=0)

        """ old implementation for comparison
        zs = []
        for query, key, value in zip(queries, keys, values):
            num_edge = key.size(0)
            query = torch.unsqueeze(query, dim=0)
            query = query.repeat(num_edge, 1)  # [num_edge*

            score = query * key
            score = score.reshape(num_edge, self.num_head, self.model_dim)
            score = torch.sum(score, dim=2)  # num_edge * num_head
            score = score / torch.sqrt(torch.Tensor((self.model_dim,))) 
            if self.pooling_op == 'softmax':
                score = torch.nn.functional.softmax(score, dim=0)  # num_edge * num_head
            elif self.pooling_op == 'relu':
                numer = torch.nn.functional.relu(score).pow(2) + self.eps
                denom = numer.sum(dim=0, keepdim=True) + self.eps
                score = numer / denom
            else:
                raise RuntimeError("No way! '{}' cannot be true".format(self.pooling_op))
            score = torch.unsqueeze(score, dim=-1)  # num_edge * num_head * 1
            value = value.reshape(num_edge, self.num_head, self.model_dim)

            z = score * value  # num_edge*num_head*model_dim
            z = torch.sum(z, dim=0)  # num_head * model_dim
            zs.append(z)
        ret_z = torch.stack(zs)
        """

        return {'z': ret_z}

    def apply_node_function(self, nodes):
        z = nodes.data['z']  # num_nodes * num_head * model_dim
        z = z.reshape(-1, self.num_head * self.model_dim)

        if self.concat_self_o:
            z = torch.cat([z, nodes.data['node_feature']], dim=1)

        out = self.WO(z)

        return {'updated_node_feature': out}
