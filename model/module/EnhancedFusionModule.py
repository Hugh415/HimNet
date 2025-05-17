import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import torch.nn as nn
from .MultiHeadAttentionFusion import MultiHeadAttentionFusion


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