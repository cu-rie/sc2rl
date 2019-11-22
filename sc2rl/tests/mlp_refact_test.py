from sc2rl.nn.MLP_refac import MultiLayerPerceptron as MLP
import torch

if __name__ == "__main__":

    torch.manual_seed(0)

    mlp = MLP(input_dimension=128,
              output_dimension=5,
              num_neurons=[128, 128],
              activation=None,
              out_activation=None,
              normalization='spectral',
              weight_init='kaiming_normal',
              dropout_probability=0.3)

    out = mlp(torch.ones(10, 128))

    print(out)