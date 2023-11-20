# Imports copied from previous code, I'm not sure I'm actually using all of them
import torch as t
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, GRU, Identity, Dropout

import torch_geometric.transforms as T
from torch_geometric.nn import Set2Set
#from torch_geometric.nn import GAT
from torch_geometric.nn import GCN
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops
from torch.nn.init import kaiming_normal_, kaiming_normal

# Two things called Linear, one in torch.nn, the other in torch_geometric
import torch_geometric as tg

import cauldronoid

import pandas as pd
from torch.optim import Adam
from time import time

from pandas import DataFrame, concat

# Just looked at the table, later should write code to automatically
# determine it
n_atom_feature = 21

# The network architecture

class Net(t.nn.Module):
    def __init__(self, num_features, dim, dropout, n_targets):
        super(Net, self).__init__()
        # Number of set2set processing steps
        # Named just in case I want to make it an argument
        steps = 2
        # Number of convolutional layers
        n_conv_layers = 2

        self.embed1 = t.nn.Linear(num_features, dim)
        self.embed2 = t.nn.Linear(dim, dim)
        self.m = Dropout(p = dropout)
        # Smaller dropout for the last layer
        self.m_small = Dropout(p = min(dropout, .2))

        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GAT
        #self.conv_layers = GAT(dim, dim, n_conv_layers,
        #                       dropout = dropout, v2 = True)
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GCN
        self.conv_layers = GCN(dim, dim, n_conv_layers, dim,
                               dropout = dropout)

        self.set2set = Set2Set(dim, processing_steps=steps)
        self.lin1 = t.nn.Linear(2 * dim, dim)
        self.lin2 = t.nn.Linear(dim, n_targets)

    def forward(self, data):
        # Two initial embedding layers
        pre_out = self.m(F.relu(self.embed1(data.x)))
        out = self.m(F.relu(self.embed2(pre_out)))

        # Graph convolution to update atom embeddings
        out = self.conv_layers(out, data.edge_index)

        # Pooling to get a molecule embedding
        out = self.set2set(out, data.batch)

        # Ouptut layers to get a prediction
        out = self.m_small(F.relu(self.lin1(out)))
        out = self.lin2(out)
        return out


def init_weights(m):
    if type(m) == t.nn.Linear or type(m) == tg.nn.dense.linear.Linear:
        kaiming_normal_(m.weight, nonlinearity = "relu", mode = "fan_out")

condition_dicts = \
    [{"name": "widewithdropoutlr1e-4", "lr": 1e-4, "dim": 100, "dropout": 0.5, "batch_size": .1}]

condition_dicts = \
    [{"name": "widewithdropoutlr1e-4", "lr": 1e-4, "dim": 100, "dropout": 0.5, "batch_size": .1}]

def new_model(n_atom_feature, condition):
    dim = condition["dim"]
    dropout = condition["dropout"]
    model = Net(n_atom_feature, condition["dim"], condition["dropout"], 1)
    model.apply(init_weights)
    return model

model = new_model(n_atom_feature, condition_dicts[0])

model.load_state_dict(t.load("full_energy_model_weights.pt"))

pred_mols = cauldronoid.files2mols(prefix = "featurized")
pred_data = [mol.get_torchgeom() for mol in pred_mols]

dataframes = []
# Breaking it into batches to avoid running out of memory
loader = DataLoader(pred_data, batch_size = 200)
    # Turning off gradient evaluation to speed it up
with t.no_grad():
    for batch in loader:
        results = model(batch)
        out_data = DataFrame(results.detach().numpy(), columns = ["z"])
        out_data["mol_id"] = batch.mol_id
        dataframes.append(out_data)

# Saving the predictions
synthetic_predictions = concat(dataframes)
synthetic_predictions.to_csv("ml_predictions.csv", index = False)
