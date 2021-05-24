import torch
import torch.nn as nn
from .canine_layer import RobertaLayer

class CanineEmbedding(nn.Module):
    def __init__(self, config, hidden_dim):
        super(CanineEmbedding, self).__init__()
        self.k = 8
        self.B = 16000
        self.dim = hidden_dim/self.k

        self.hash_embedding = nn.ModuleList([])
        for i in range(self.k):
            ha = nn.Embedding(self.B, int(self.dim))
            self.hash_embedding.append(ha)
        
        self.local_transformer = RobertaLayer(config)
        config.down_sample_rate = int(2048/128)
        # input: N, C_in, L_in; output: N, C_out, L_out
        self.conv_downsample = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=config.down_sample_rate, stride=config.down_sample_rate)

        self.conv_upsample = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=config.down_sample_rate, stride=config.down_sample_rate)
        self.transformer = RobertaLayer(config)
        self.config = config

    def mask_embedding(self):
        return self.ngram_proj.weight[:,self.tokenizer.mask_token_id]

    def init_weight(self):
        nn.init.normal_(self.ngram_proj.weight, mean=0, std=self.config.initializer_range*0.5)
        if self.latent_mat is not None:
            nn.init.normal_(self.latent_mat, mean=0, std=self.config.initializer_range*0.5)
        nn.init.constant_(self.ngram_proj.weight[self.padding_idx], 0.0)

    def forward(self, x):
        print(x)
        exit(0)
        # x: [batch_size, len]
        # step1: hash embedding
        #
        # step2: local transformer
        #
        # step3: conv downsample

        return token_emb

    def forward_up(self, x, max_len, lang_pair=None):
        pass

    @property
    def weight(self):
        return None


