import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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