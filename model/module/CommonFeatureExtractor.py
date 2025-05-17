import warnings
warnings.filterwarnings("ignore", message="It is not recommended to directly access")
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        maccs = fp_features[:, self.atom_pairs_dim:self.atom_pairs_dim + self.maccs_dim]
        morgan_bits = fp_features[:, self.atom_pairs_dim + self.maccs_dim:
                                     self.atom_pairs_dim + self.maccs_dim + self.morgan_bits_dim]
        morgan_counts = fp_features[:, self.atom_pairs_dim + self.maccs_dim + self.morgan_bits_dim:
                                       self.atom_pairs_dim + self.maccs_dim + self.morgan_bits_dim + self.morgan_counts_dim]
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
                for j in range(i + 1, 5):
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