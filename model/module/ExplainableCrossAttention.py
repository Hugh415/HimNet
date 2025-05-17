import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplainableCrossAttention(nn.Module):
    """
    Interpretable cross-attention that clearly distinguishes interactions between different types of nodes
    """

    def __init__(self, hidden_size, num_heads=8, dropout=0.1, device=None):
        super(ExplainableCrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert self.head_dim * num_heads == hidden_size, "hidden_size必须能被num_heads整除"

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.scale = self.head_dim ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self._reset_parameters()

        self.attention_weights = {}

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, atom_features, a_scope, mol_atom_num=None):
        """
        Applies explainable cross-attention to nodes in hierarchical molecular graphs

        Args:
            atom_features: Features of all nodes with shape [num_nodes, hidden_size]
            a_scope: Index ranges for molecules in the batch, formatted as [(start_idx, num_nodes), ...]
            mol_atom_num: Number of atoms per molecule, used to distinguish node types

        Returns:
            enhanced_features: Node features enhanced by cross-attention [num_nodes, hidden_size]
            attention_patterns: Dictionary of attention patterns containing interactions between different node types
        """
        enhanced_features = torch.zeros_like(atom_features)

        attention_patterns = {
            'atom-atom': [],
            'atom-motif': [],
            'motif-motif': [],
            'atom-graph': [],
            'motif-graph': []
        }

        for idx, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                continue

            mol_features = atom_features[a_start:a_start + a_size]  # [a_size, hidden_size]

            num_atoms = mol_atom_num[idx] if mol_atom_num is not None else a_size // 3

            num_graph = 1
            num_motifs = a_size - num_atoms - num_graph

            node_types = torch.zeros(a_size, device=self.device)
            if num_atoms > 0:
                node_types[:num_atoms] = 0
            if num_motifs > 0:
                node_types[num_atoms:num_atoms + num_motifs] = 1
            if num_graph > 0:
                node_types[num_atoms + num_motifs:] = 2

            atom_mask = (node_types == 0)
            motif_mask = (node_types == 1)
            graph_mask = (node_types == 2)

            residual = mol_features

            mol_features = mol_features.unsqueeze(0)  # [1, a_size, hidden_size]

            q = self.q_proj(mol_features)  # [1, a_size, hidden_size]
            k = self.k_proj(mol_features)  # [1, a_size, hidden_size]
            v = self.v_proj(mol_features)  # [1, a_size, hidden_size]

            batch_size, seq_len, _ = mol_features.size()
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                     2)  # [1, num_heads, a_size, head_dim]
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                     2)  # [1, num_heads, a_size, head_dim]
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                     2)  # [1, num_heads, a_size, head_dim]

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [1, num_heads, a_size, a_size]

            attn_weights = F.softmax(attn_scores, dim=-1)  # [1, num_heads, a_size, a_size]

            attn_map = attn_weights[0, 0].clone().to(self.device)  # [a_size, a_size]

            mol_id = idx

            if atom_mask.sum() > 0:
                if atom_mask.sum() > 0:
                    aa_mask = torch.outer(atom_mask, atom_mask)
                    aa_attn = attn_map.masked_select(aa_mask).reshape(atom_mask.sum().item(), atom_mask.sum().item())
                    attention_patterns['atom-atom'].append((mol_id, aa_attn.cpu().detach()))

                if motif_mask.sum() > 0:
                    am_mask = torch.outer(atom_mask, motif_mask)
                    am_attn = attn_map.masked_select(am_mask).reshape(atom_mask.sum().item(), motif_mask.sum().item())
                    attention_patterns['atom-motif'].append((mol_id, am_attn.cpu().detach()))

                if graph_mask.sum() > 0:
                    ag_mask = torch.outer(atom_mask, graph_mask)
                    ag_attn = attn_map.masked_select(ag_mask).reshape(atom_mask.sum().item(), graph_mask.sum().item())
                    attention_patterns['atom-graph'].append((mol_id, ag_attn.cpu().detach()))

            if motif_mask.sum() > 0:
                if motif_mask.sum() > 0:
                    mm_mask = torch.outer(motif_mask, motif_mask)
                    mm_attn = attn_map.masked_select(mm_mask).reshape(motif_mask.sum().item(), motif_mask.sum().item())
                    attention_patterns['motif-motif'].append((mol_id, mm_attn.cpu().detach()))

                if graph_mask.sum() > 0:
                    mg_mask = torch.outer(motif_mask, graph_mask)
                    mg_attn = attn_map.masked_select(mg_mask).reshape(motif_mask.sum().item(), graph_mask.sum().item())
                    attention_patterns['motif-graph'].append((mol_id, mg_attn.cpu().detach()))

            self.attention_weights[f'molecule_{mol_id}'] = {
                'weights': attn_map.detach().cpu().numpy(),
                'node_types': node_types.cpu().numpy(),
                'num_atoms': num_atoms,
                'num_motifs': num_motifs,
                'num_graph': num_graph
            }

            attn_weights = self.dropout(attn_weights)

            context = torch.matmul(attn_weights, v)  # [1, num_heads, a_size, head_dim]
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                                self.hidden_size)  # [1, a_size, hidden_size]

            output = self.out_proj(context)  # [1, a_size, hidden_size]

            output = output.squeeze(0)  # [a_size, hidden_size]

            output = self.layer_norm(output + residual)

            enhanced_features[a_start:a_start + a_size] = output

        return enhanced_features, attention_patterns

    def visualize_attention(self, molecule_idx=None, save_path=None):
        """
        Visualizes attention weights for specified molecules.

        Args:
            molecule_idx (int, optional): Index of the molecule to visualize. If None, visualizes all molecules.
            save_path (str, optional): Path to save the visualization image. If None, displays the image directly.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not self.attention_weights:
            print("Attention weighting data without visualization")
            return

        keys = [f'molecule_{molecule_idx}'] if molecule_idx is not None else self.attention_weights.keys()

        for key in keys:
            if key not in self.attention_weights:
                print(f"Can't find the attention weight of the molecule {key}.")
                continue

            data = self.attention_weights[key]
            weights = data['weights']
            node_types = data['node_types']
            num_atoms = data['num_atoms']
            num_motifs = data['num_motifs']

            labels = []
            for i in range(len(node_types)):
                if i < num_atoms:
                    labels.append(f'A{i}')
                elif i < num_atoms + num_motifs:
                    labels.append(f'M{i - num_atoms}')
                else:
                    labels.append('G')

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(weights, cmap='viridis')

            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            cbar = ax.figure.colorbar(im, ax=ax)

            ax.set_title(f"weighting of attention - {key}")

            if num_atoms > 0:
                ax.axhline(y=num_atoms - 0.5, color='r', linestyle='-', alpha=0.3)
                ax.axvline(x=num_atoms - 0.5, color='r', linestyle='-', alpha=0.3)

            if num_motifs > 0:
                ax.axhline(y=num_atoms + num_motifs - 0.5, color='g', linestyle='-', alpha=0.3)
                ax.axvline(x=num_atoms + num_motifs - 0.5, color='g', linestyle='-', alpha=0.3)

            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = ax.text(j, i, f'{weights[i, j]:.2f}',
                                   ha="center", va="center", color="w" if weights[i, j] > 0.5 else "black",
                                   fontsize=7)

            plt.tight_layout()

            if save_path:
                plt.savefig(f"{save_path}_{key}.png", dpi=300, bbox_inches='tight')
            else:
                plt.show()

            plt.close()
