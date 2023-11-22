import torch
import torch_geometric

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
ParticleNet Implementation
'''

class ParticleStaticEdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ParticleStaticEdgeConv, self).__init__(aggr='max')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, out_channels[0], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[0]), 
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[0], out_channels[1], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[1], out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index, k):
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, edge_index, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)

        out_mlp = self.mlp(tmp)

        return out_mlp

    def update(self, aggr_out):
        return aggr_out



class ParticleDynamicEdgeConv(ParticleStaticEdgeConv):
    def __init__(self, in_channels, out_channels, k=7):
        super(ParticleDynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k
        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()

    def forward(self, pts, fts, batch=None):
        edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False, flow=self.flow)
        aggrg = super(ParticleDynamicEdgeConv, self).forward(fts, edges, self.k)
        x = self.skip_mlp(fts)
        out = torch.add(aggrg, x)
        return self.act(out)



class ParticleNet(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        previous_output_shape = config['input_size']

        self.input_bn = torch_geometric.nn.BatchNorm(config['input_size'])

        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(config['conv_params']):
            K = layer_param[0]
            channels = layer_param[1]
            self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=K).to(DEVICE))
            previous_output_shape = channels[-1]



        self.fc_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(config['fc_params']):
            drop_rate, units = layer_param[0], layer_param[1]
            seq = torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p=drop_rate),
                torch.nn.ReLU()
            ).to(DEVICE)
            self.fc_process.append(seq)
            previous_output_shape = units


        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, config['output_size'])
        self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        fts = self.input_bn(batch.x)
        pts = batch.pos

        for idx, layer in enumerate(self.conv_process):
          fts = layer(pts, fts, batch.batch)
          pts = fts

        x = torch_geometric.nn.global_mean_pool(fts, batch.batch)

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)
        x = self.output_activation(x)
        return x
    


if __name__ == '__main__':
    settings = {
        "conv_params": [
            {'k': 16, 'C': (64, 64, 64)},
            {'k': 16, 'C': (128, 128, 128)},
            {'k': 16, 'C': (256, 256, 256)},
        ],
        "fc_params": [
            {'p': 0.1, 'units': 256}
        ],
        "input_size": 4,
        "output_size": 2,
    }

    model = ParticleNet(settings)

    print(model)