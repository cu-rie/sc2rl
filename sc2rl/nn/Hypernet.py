from .MultiLayerPerceptron import MultiLayerPerceptron as MLP


class HyperNetwork(MLP):

    def __init__(self, num_rows, num_cols, **kwargs):
        self.num_rows = num_rows
        self.num_cols = num_cols
        super(HyperNetwork, self).__init__(**kwargs)

    def forward(self, x):
        out = super(HyperNetwork, self).forward(x)
        out = out.unsqueeze(-1)
        out = out.view(-1, self.num_rows, self.num_cols)
        return out
