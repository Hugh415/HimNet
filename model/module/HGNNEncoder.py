import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            key = self.W_atom_k(mol_f_atoms)  # a_size x atom_fdim
            value = self.W_atom_v(mol_f_atoms)  # a_size x atom_fdim

            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.atom_fdim)

            attention_weights = F.softmax(scores, dim=-1)  # a_size x a_size

            attention_output = torch.matmul(attention_weights, value)  # a_size x atom_fdim

            curr_enhanced = self.layer_norm_atom(mol_f_atoms + attention_output)  # a_size x atom_fdim

            enhanced_atoms[a_start:a_start + a_size] = curr_enhanced

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