import dgl


if __name__ == "__main__":
    follows_g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows')
    devs_g = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game')
    hetero_g = dgl.hetero_from_relations([follows_g, devs_g])
    homo_g = dgl.to_homo(hetero_g)

    hetero_g_2 = dgl.to_hetero(homo_g, hetero_g.ntypes, hetero_g.etypes)

    print(hetero_g)
    print(hetero_g_2)
    print("here")