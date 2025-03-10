import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv, GCNConv, GINConv


class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GCN(torch.nn.Module):
    """Graph Convolutional Network"""

    def __init__(self, dim_in, dim_h, dim_out, num_layers=5, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dim_in, dim_h))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(dim_h, dim_h))
        self.convs.append(GCNConv(dim_h, dim_out))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, dim_in, dim_h, dim_out, num_layers=5, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(dim_in, dim_h))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(dim_h, dim_h))
        self.convs.append(GATConv(dim_h, dim_out))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GIN(torch.nn.Module):
    """GIN"""

    def __init__(self, dim_in, dim_h, dim_out, num_layers=5, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        self.convs.append(GINConv(Sequential(Linear(dim_in, dim_h),
                                             BatchNorm1d(dim_h), ReLU(),
                                             Linear(dim_h, dim_h))))

        for _ in range(num_layers - 2):
            self.convs.append(GINConv(
                Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                           Linear(dim_h, dim_h))))
        self.output_layer = torch.nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):  # -1 to exclude the output layer
            x = self.convs[i](x, edge_index)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_layer(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        # Create the first layer
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        # Add dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Create additional hidden layers
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        out = self.output_layer(x)
        return out
