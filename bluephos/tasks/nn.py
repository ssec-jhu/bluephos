import pandas as pd
import torch as t
import torch.nn.functional as F
import bluephos.modules.log_config as log_config
from dplutils.pipeline import PipelineTask
from torch.nn import Dropout, Linear
from torch.nn.init import kaiming_normal_
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCN, Set2Set

from bluephos.modules.sdf2feature import feature_create

# Setup logging and get a logger instance
logger = log_config.setup_logging(__name__)


class Net(t.nn.Module):
    def __init__(self, num_features, dim, dropout, n_targets):
        super(Net, self).__init__()
        steps = 2
        # Number of convolutional layers
        n_conv_layers = 2

        self.embed1 = Linear(num_features, dim)
        self.embed2 = Linear(dim, dim)
        self.m = Dropout(p=dropout)
        self.m_small = Dropout(p=min(dropout, 0.2))

        self.conv_layers = GCN(dim, dim, n_conv_layers, dim, dropout=dropout)

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
    if isinstance(m, t.nn.Linear):
        kaiming_normal_(m.weight, nonlinearity="relu", mode="fan_out")


def new_model(n_atom_feature, condition):
    model = Net(n_atom_feature, condition["dim"], condition["dropout"], 1)
    model.apply(init_weights)
    return model


def apply_nn(feature_df: pd.DataFrame, model_weights) -> pd.DataFrame:
    # Load the pre-trained model
    condition_dicts = [
        {
            "name": "widewithdropoutlr1e-4",
            "lr": 1e-4,
            "dim": 100,
            "dropout": 0.5,
            "batch_size": 0.1,
        }
    ]

    pred_mols = feature_df["Molecule"].tolist()
    pred_data = [mol.get_torch_geom_data() for mol in pred_mols]

    # Infer the number of atom features from the input data
    if pred_data:
        n_atom_feature = pred_data[0].x.shape[1]
    else:
        raise ValueError("No input data found to infer atom features.")

    model = new_model(n_atom_feature, condition_dicts[0])
    model.load_state_dict(t.load(model_weights))

    model.eval()

    # Make predictions
    dataframes = []

    loader = DataLoader(pred_data)

    # Turning off gradient evaluation to speed it up
    with t.no_grad():
        for batch in loader:
            results = model(batch)
            out_data = pd.DataFrame(results.detach().numpy(), columns=["z"])
            out_data["mol_id"] = batch.mol_id
            logger.info(f"Processed molecule ID: {batch.mol_id}")
            dataframes.append(out_data)

    # Saving the predictions
    nn_predictions = pd.concat(dataframes)

    return nn_predictions


def nn(df: pd.DataFrame, element_features, train_stats, model_weights) -> pd.DataFrame:
    df_structure = df[["ligand_identifier", "structure"]].dropna(subset=["structure"])
    feature_df = feature_create(df_structure, element_features, train_stats)
    nn_score_df = apply_nn(feature_df, model_weights)
    score_mapping = nn_score_df.set_index("mol_id")["z"]
    df["z"] = df["ligand_identifier"].map(score_mapping)
    logger.info("NN task complete")
    return df


NNTask = PipelineTask(
    "nn",
    nn,
    context_kwargs={
        "element_features": "element_features",
        "train_stats": "train_stats",
        "model_weights": "model_weights",
    },
    batch_size=200,
)
