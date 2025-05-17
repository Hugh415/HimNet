import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import torch
import torch.nn as nn
from dataset import BatchMolGraph
from .HGNNEncoder import HGNNEncoder
from .ExplainableCrossAttention import ExplainableCrossAttention

class ExplainableCrossAttentionEncoder(nn.Module):
    """
    Enhancing encoders for molecular representation using interpretable cross-attention mechanisms
    """

    def __init__(self, atom_fdim, bond_fdim, hidden_size, depth, device):
        super(ExplainableCrossAttentionEncoder, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.atom_fdim = atom_fdim

        self.atom_proj = nn.Linear(atom_fdim, hidden_size)

        self.mpnn_encoder = HGNNEncoder(atom_fdim, bond_fdim, hidden_size, depth, device)

        self.cross_attention = ExplainableCrossAttention(
            hidden_size=hidden_size,
            num_heads=8,
            dropout=0.1,
            device=device
        )

        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

    def forward(self, mol_graph, return_attention=False):
        """
        Forward pass with explainable cross-attention for molecular representation enhancement.

        Args:
            mol_graph (BatchMolGraph): Input batch molecular graph object
            return_attention (bool): Whether to return attention patterns (default: False)

        Returns:
            tuple: Contains:
                - enhanced_mol_vecs (Tensor): Enhanced molecular representation vectors [batch_size, hidden_size]
                - attention_patterns (dict, optional): Attention patterns if return_attention=True. None otherwise.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(
            self.device), b2a.to(self.device), b2revb.to(self.device)

        base_mol_vecs = self.mpnn_encoder(mol_graph)

        atom_hiddens = self.atom_proj(f_atoms)  # [num_atoms, hidden_size]

        enhanced_atom_features, attention_patterns = self.cross_attention(
            atom_hiddens,
            a_scope,
            mol_graph.mol_atom_num if hasattr(mol_graph, 'mol_atom_num') else None
        )

        enhanced_mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                enhanced_mol_vecs.append(self.cached_zero_vector)
            else:
                mol_atom_features = enhanced_atom_features[a_start:a_start + a_size]

                mol_vec = mol_atom_features.sum(dim=0) / a_size
                enhanced_mol_vecs.append(mol_vec)

        enhanced_mol_vecs = torch.stack(enhanced_mol_vecs, dim=0)  # [batch_size, hidden_size]

        final_mol_vecs = self.output_layer(enhanced_mol_vecs + base_mol_vecs)
        final_mol_vecs = self.act_func(final_mol_vecs)
        final_mol_vecs = self.dropout(final_mol_vecs)

        if return_attention:
            return final_mol_vecs, attention_patterns

        return final_mol_vecs

    def visualize_attention(self, molecule_idx=None, save_path=None):
        """
        Visualizes attention weights for specified molecules in the batch.

        Parameters:
            molecule_idx (int, optional): Index of the target molecule to visualize.
                If None, visualizes attention for all molecules in the batch.
            save_path (str, optional): Path to save the generated visualization.
                If None, displays the plot interactively instead of saving.
        """
        self.cross_attention.visualize_attention(molecule_idx, save_path)
