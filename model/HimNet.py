import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP
from dataset import BatchMolGraph
from module.ExplainableCrossAttentionEncoder import ExplainableCrossAttentionEncoder
from module.EnhancedFusionModule import EnhancedFusionModule
from module.CommonFeatureExtractor import CommonFeatureExtractor


class HimNet(nn.Module):
    def __init__(self,
                 data_name,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 fp_fdim: int = None,
                 hidden_size=256,
                 depth=5,
                 device='cpu',
                 out_dim=2,
                 num_heads=4,
                 #  fusion_type='concat'):
                 fusion_type='add'):
        super(HimNet, self).__init__()
        self.data_name = data_name
        self.device = device
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.fp_fdim = fp_fdim

        self.encoder = ExplainableCrossAttentionEncoder(self.atom_fdim, self.bond_fdim, hidden_size, depth, device)

        self.fp_encoder = CommonFeatureExtractor(fp_fdim, hidden_size)

        self.graph_proj_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.feature_fusion = EnhancedFusionModule(
            feat_views=2,
            feat_dim=hidden_size,
            num_heads=num_heads,
            fusion_type=fusion_type,
            device=device
        )

        self.mlp = MLP([hidden_size, hidden_size, out_dim], dropout=0.1)

    def forward(self, batch, training=False):
        mol_batch = BatchMolGraph(batch.smi, atom_fdim=self.atom_fdim, bond_fdim=self.bond_fdim,
                                  fp_fdim=self.fp_fdim, data_name=self.data_name)

        ligand_x = self.encoder.forward(mol_batch)

        if training:
            fp_x, fp_projections, fp_view_projections = self.fp_encoder(
                mol_batch.fp_x.to(self.device).to(torch.float32), training=True
            )
            graph_projections = self.graph_proj_head(ligand_x)
            graph_projections = F.normalize(graph_projections, dim=1)

            self.contrastive_data = {
                'graph_proj': graph_projections,
                'fp_proj': fp_projections,
                'fp_view_projs': fp_view_projections
            }
        else:
            fp_x = self.fp_encoder(mol_batch.fp_x.to(self.device).to(torch.float32))

        fused_x = self.feature_fusion(torch.stack([ligand_x, fp_x], dim=0))

        x = self.mlp(fused_x)
        return x

    def compute_contrastive_loss(self, temperature=0.1):
        """Calculating Comparative Learning Losses"""
        if not hasattr(self, 'contrastive_data'):
            return torch.tensor(0.0, device=self.device)

        graph_proj = self.contrastive_data['graph_proj']
        fp_proj = self.contrastive_data['fp_proj']

        batch_size = graph_proj.size(0)

        sim_matrix = torch.mm(graph_proj, fp_proj.t()) / temperature

        labels = torch.arange(batch_size, device=self.device)
        loss_graph_fp = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)
        loss_graph_fp = loss_graph_fp / 2

        fp_view_projs = self.contrastive_data['fp_view_projs']
        fp_keys = list(fp_view_projs.keys())

        loss_fp_views = 0
        count = 0

        for i in range(len(fp_keys)):
            for j in range(i + 1, len(fp_keys)):
                view_i = fp_view_projs[fp_keys[i]]
                view_j = fp_view_projs[fp_keys[j]]

                view_sim = torch.mm(view_i, view_j.t()) / temperature

                view_loss = F.cross_entropy(view_sim, labels) + F.cross_entropy(view_sim.t(), labels)
                loss_fp_views += view_loss
                count += 2

        if count > 0:
            loss_fp_views = loss_fp_views / count

        total_contrastive_loss = loss_graph_fp + 0.5 * loss_fp_views

        return total_contrastive_loss

    def get_batch_mol_graph(self, batch):
        """Creates and returns a BatchMolGraph object for visualization"""
        return BatchMolGraph(batch.smi, atom_fdim=self.atom_fdim, bond_fdim=self.bond_fdim,
                             fp_fdim=self.fp_fdim, data_name=self.data_name)