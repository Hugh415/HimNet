import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.models import MLP
from dataset import BatchMolGraph

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


class HGNNEncoder(nn.Module):
    def __init__(self, atom_fdim, bond_fdim, hidden_size, depth, device, num_heads=4):
        super(HGNNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.device = device
        self.bias = False
        self.num_heads = num_heads
        
        self.dropout_layer = nn.Dropout(p=0.1)
        
        self.act_func = nn.ReLU()
        
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        self.W_i = nn.Linear(self.bond_fdim, self.hidden_size, bias=self.bias)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)
        self.W_a = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.W_b = nn.Linear(self.hidden_size, self.hidden_size)
        
        
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        
        self.W_a = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        
        self.W_alpha = nn.Linear(self.hidden_size * 2, 1, bias=True)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        self.W_atom_q = nn.Linear(atom_fdim, atom_fdim, bias=self.bias)
        self.W_atom_k = nn.Linear(atom_fdim, atom_fdim, bias=self.bias)
        self.W_atom_v = nn.Linear(atom_fdim, atom_fdim, bias=self.bias)
        self.W_atom_proj = nn.Identity()
        self.layer_norm_atom = nn.LayerNorm(atom_fdim)
        
        if num_heads > 1:
            self.W_atom_multihead_out = nn.Linear(hidden_size * num_heads, hidden_size)

        if hidden_size != atom_fdim:
            self.W_atom_back_proj = nn.Linear(hidden_size, atom_fdim, bias=self.bias)

    def forward(self, mol_graph):
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(
            self.device), b2a.to(self.device), b2revb.to(self.device)

        enhanced_atoms = f_atoms.clone()
        
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size <= 1:
                continue
                
            mol_f_atoms = f_atoms.narrow(0, a_start, a_size)  # a_size x atom_fdim
            
            query = self.W_atom_q(mol_f_atoms)  # a_size x atom_fdim
            key = self.W_atom_k(mol_f_atoms)    # a_size x atom_fdim
            value = self.W_atom_v(mol_f_atoms)  # a_size x atom_fdim
            
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.atom_fdim)
            
            attention_weights = F.softmax(scores, dim=-1)  # a_size x a_size
            
            attention_output = torch.matmul(attention_weights, value)  # a_size x atom_fdim
            
            curr_enhanced = self.layer_norm_atom(mol_f_atoms + attention_output)  # a_size x atom_fdim
            
            enhanced_atoms[a_start:a_start+a_size] = curr_enhanced
        
        f_atoms = enhanced_atoms

        inputs = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(inputs)  # num_bonds x hidden_size

        for depth in range(self.depth - 1):
            nei_a_message = index_select_ND(message, a2b)
            a_message = nei_a_message.sum(dim=1)
            rev_message = message[b2revb]
            dmpnn_message = a_message[b2a] - rev_message

            dmpnn_message = self.W_h(dmpnn_message)

            Q = self.W_q(message)
            K = self.W_k(message)
            V = self.W_v(message)
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size)
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_message = torch.matmul(attention_weights, V)
            attention_message = self.W_a(attention_message)

            alpha = torch.sigmoid(self.W_alpha(torch.cat([dmpnn_message, attention_message], dim=-1)))
            combined_message = alpha * dmpnn_message + (1 - alpha) * attention_message

            message = self.act_func(inputs + combined_message)
            message = self.dropout_layer(message)
            

        nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
     
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)

                att_w = torch.matmul(self.W_a(cur_hiddens), cur_hiddens.t())
                att_w = F.softmax(att_w, dim=1)
                att_hiddens = torch.matmul(att_w, cur_hiddens)
                att_hiddens = self.act_func(self.W_b(att_hiddens))
                att_hiddens = self.dropout_layer(att_hiddens)
                mol_vec = (cur_hiddens + att_hiddens)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden

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
                
            mol_features = atom_features[a_start:a_start+a_size]  # [a_size, hidden_size]
            
            num_atoms = mol_atom_num[idx] if mol_atom_num is not None else a_size // 3
            
            num_graph = 1
            num_motifs = a_size - num_atoms - num_graph
            
            node_types = torch.zeros(a_size, device=self.device)
            if num_atoms > 0:
                node_types[:num_atoms] = 0
            if num_motifs > 0:
                node_types[num_atoms:num_atoms+num_motifs] = 1
            if num_graph > 0:
                node_types[num_atoms+num_motifs:] = 2
            
            atom_mask = (node_types == 0)
            motif_mask = (node_types == 1)
            graph_mask = (node_types == 2)
            
            residual = mol_features
            
            mol_features = mol_features.unsqueeze(0)  # [1, a_size, hidden_size]
            
            q = self.q_proj(mol_features)  # [1, a_size, hidden_size]
            k = self.k_proj(mol_features)  # [1, a_size, hidden_size]
            v = self.v_proj(mol_features)  # [1, a_size, hidden_size]
            
            batch_size, seq_len, _ = mol_features.size()
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [1, num_heads, a_size, head_dim]
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [1, num_heads, a_size, head_dim]
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [1, num_heads, a_size, head_dim]
            
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
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)  # [1, a_size, hidden_size]
            
            output = self.out_proj(context)  # [1, a_size, hidden_size]
            
            output = output.squeeze(0)  # [a_size, hidden_size]
            
            output = self.layer_norm(output + residual)
            
            enhanced_features[a_start:a_start+a_size] = output
            
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
                    labels.append(f'M{i-num_atoms}')
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
                ax.axhline(y=num_atoms-0.5, color='r', linestyle='-', alpha=0.3)
                ax.axvline(x=num_atoms-0.5, color='r', linestyle='-', alpha=0.3)
            
            if num_motifs > 0:
                ax.axhline(y=num_atoms+num_motifs-0.5, color='g', linestyle='-', alpha=0.3)
                ax.axvline(x=num_atoms+num_motifs-0.5, color='g', linestyle='-', alpha=0.3)
            
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
                mol_atom_features = enhanced_atom_features[a_start:a_start+a_size]
                
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

class WeightFusion(nn.Module):
    def __init__(self, feat_views, feat_dim, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightFusion, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.weight = Parameter(torch.empty((1, 1, feat_views), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(int(feat_dim), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:

        return sum([inputs[i] * weight for i, weight in enumerate(self.weight[0][0])]) + self.bias

class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(MultiHeadAttentionFusion, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        batch_size = inputs.size(1)
        features = inputs.transpose(0, 1)  # [batch_size, num_views, hidden_size]
        
        query = self.query(features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        output = self.out_proj(context).mean(dim=1)
        
        return output

class EnhancedFusionModule(nn.Module):
    def __init__(self, feat_views, feat_dim, num_heads=4, fusion_type='concat', device=None):
        super(EnhancedFusionModule, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.fusion_type = fusion_type
        
        self.attn_fusion = MultiHeadAttentionFusion(feat_dim, num_heads=num_heads)
        
        if fusion_type == 'concat':
            self.projection = nn.Linear(feat_dim * 2, feat_dim)
        
    def forward(self, inputs):
        feature_B = self.attn_fusion(inputs)
        return feature_B

'''
Molecular Fingerprint Common Feature Extraction Module
'''
class CommonFeatureExtractor(nn.Module):
    def __init__(self, fp_dim, hidden_size, similarity_threshold=0.6):
        super(CommonFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.similarity_threshold = similarity_threshold
        
        self.atom_pairs_dim = 2048
        self.maccs_dim = 167
        self.morgan_bits_dim = 2048
        self.morgan_counts_dim = 2048
        self.pharmacophore_dim = 27
        
        self.fp_encoders = nn.ModuleDict({
            'atom_pairs': nn.Sequential(
                nn.Linear(self.atom_pairs_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, hidden_size)
            ),
            'maccs': nn.Sequential(
                nn.Linear(self.maccs_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, hidden_size)
            ),
            'morgan_bits': nn.Sequential(
                nn.Linear(self.morgan_bits_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, hidden_size)
            ),
            'morgan_counts': nn.Sequential(
                nn.Linear(self.morgan_counts_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, hidden_size)
            ),
            'pharmacophore': nn.Sequential(
                nn.Linear(self.pharmacophore_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, hidden_size)
            )
        })
        
        self.commonality_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.enhancement_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.weight_generator = nn.Linear(hidden_size * 5, 5)
        
        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, fp_features, training=False):
        batch_size = fp_features.size(0)
        
        atom_pairs = fp_features[:, :self.atom_pairs_dim]
        maccs = fp_features[:, self.atom_pairs_dim:self.atom_pairs_dim+self.maccs_dim]
        morgan_bits = fp_features[:, self.atom_pairs_dim+self.maccs_dim:
                                 self.atom_pairs_dim+self.maccs_dim+self.morgan_bits_dim]
        morgan_counts = fp_features[:, self.atom_pairs_dim+self.maccs_dim+self.morgan_bits_dim:
                                   self.atom_pairs_dim+self.maccs_dim+self.morgan_bits_dim+self.morgan_counts_dim]
        pharmacophore = fp_features[:, -self.pharmacophore_dim:]
        
        encoded_fps = {
            'atom_pairs': self.fp_encoders['atom_pairs'](atom_pairs),
            'maccs': self.fp_encoders['maccs'](maccs),
            'morgan_bits': self.fp_encoders['morgan_bits'](morgan_bits),
            'morgan_counts': self.fp_encoders['morgan_counts'](morgan_counts),
            'pharmacophore': self.fp_encoders['pharmacophore'](pharmacophore)
        }
        
        all_fp_common_features = []
        fp_keys = list(encoded_fps.keys())
        
        for b in range(batch_size):
            sample_fps = torch.stack([encoded_fps[k][b] for k in fp_keys])  # [5, hidden_size]
            
            sample_fps_norm = F.normalize(sample_fps, dim=1)
            similarity_matrix = torch.mm(sample_fps_norm, sample_fps_norm.t())  # [5, 5]
            
            mask = torch.eye(5, device=fp_features.device)
            similarity_matrix = similarity_matrix * (1 - mask)
            
            pairwise_similarities = []
            for i in range(4):
                for j in range(i+1, 5):
                    pairwise_similarities.append((i, j, similarity_matrix[i, j]))
            
            pairwise_similarities.sort(key=lambda x: x[2], reverse=True)
            
            common_features = []
            pair_weights = []
            
            for i, j, sim in pairwise_similarities:
                if sim > self.similarity_threshold:
                    fp1, fp2 = sample_fps[i], sample_fps[j]
                    fp1_norm, fp2_norm = sample_fps_norm[i], sample_fps_norm[j]
                    
                    element_sim = fp1_norm * fp2_norm
                    
                    high_sim_mask = (element_sim > self.similarity_threshold).float()
                    common_feature = ((fp1 + fp2) / 2) * high_sim_mask
                    
                    common_features.append(common_feature)
                    pair_weights.append(sim)
            
            if not common_features:
                sample_common = torch.mean(sample_fps, dim=0)
            else:
                pair_weights = torch.tensor(pair_weights, device=fp_features.device)
                pair_weights = F.softmax(pair_weights, dim=0)
                common_features = torch.stack(common_features)
                sample_common = torch.sum(common_features * pair_weights.unsqueeze(1), dim=0)
            
            all_fp_common_features.append(sample_common)
        
        common_features = torch.stack(all_fp_common_features)  # [batch_size, hidden_size]
        
        all_fps_concat = torch.cat([encoded_fps[k] for k in fp_keys], dim=1)  # [batch_size, hidden_size*5]
        fp_weights = F.softmax(self.weight_generator(all_fps_concat), dim=1)  # [batch_size, 5]
        
        weighted_fp_sum = torch.zeros(batch_size, self.hidden_size, device=fp_features.device)
        for i, k in enumerate(fp_keys):
            weighted_fp_sum += encoded_fps[k] * fp_weights[:, i].unsqueeze(1)
        
        enhancement_factors = self.enhancement_layer(common_features)
        enhanced_common = common_features * enhancement_factors
        
        fused_features = self.fusion_layer(torch.cat([weighted_fp_sum, enhanced_common], dim=1))
        
        if training:
            projections = self.projection_head(fused_features)
            projections = F.normalize(projections, dim=1)
            
            fp_projections = {}
            for k in fp_keys:
                proj = self.projection_head(encoded_fps[k])
                fp_projections[k] = F.normalize(proj, dim=1)
            
            return fused_features, projections, fp_projections
        
        return fused_features


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
            for j in range(i+1, len(fp_keys)):
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
