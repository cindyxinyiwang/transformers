import torch
import torch.nn as nn
from .canine_layer import RobertaLayer

# Support up to 16 hash functions.
_PRIMES = [
    31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223
]

class CanineEmbedding(nn.Module):
    def __init__(self, config):
        super(CanineEmbedding, self).__init__()
        self.k = 8
        self.B = 16000
        self.dim = config.hidden_size/self.k

        self.hash_embedding = nn.ModuleList([])
        for i in range(self.k):
            ha = nn.Embedding(self.B, int(self.dim))
            self.hash_embedding.append(ha)

        self.local_transformer = RobertaLayer(config)
        config.down_sample_rate = int(2048/512)
        self.down_sample_rate = config.down_sample_rate
        # input: N, C_in, L_in; output: N, C_out, L_out
        self.conv_downsample = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=config.down_sample_rate, stride=config.down_sample_rate)

        config.upsample_kernel_size = 4
        self.conv_upsample = nn.Conv1d(in_channels=config.hidden_size*2, out_channels=config.hidden_size, kernel_size=config.upsample_kernel_size, stride=1)
        self.transformer = RobertaLayer(config)
        self.config = config
        self.pad_token_id = config.pad_token_id

        # position and token embedding
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )



    def mask_embedding(self):
        return self.ngram_proj.weight[:,self.tokenizer.mask_token_id]

    def init_weight(self):
        nn.init.normal_(self.ngram_proj.weight, mean=0, std=self.config.initializer_range*0.5)
        if self.latent_mat is not None:
            nn.init.normal_(self.latent_mat, mean=0, std=self.config.initializer_range*0.5)
        nn.init.constant_(self.ngram_proj.weight[self.padding_idx], 0.0)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # x: [batch_size, len]
        # mask 0 for nonpad and 1 for pad
        mask = input_ids.eq(self.pad_token_id)
        if mask.any():
            mask = mask.float() * -10000.0
        else:
            mask = None
        # step1: hash embedding
        hash_bucket_tensors = self._hash_bucket_tensors(input_ids, self.k, self.B)
        embedding_shards = []
        for i, hash_bucket_ids in enumerate(hash_bucket_tensors):
            shard_embedding = self.hash_embedding[i](hash_bucket_ids)
            embedding_shards.append(shard_embedding)
        hashed_embedding = torch.cat(embedding_shards, dim=-1)
        embedding = self.char_embedding_postprocess(inputs_embeds=hashed_embedding, token_type_ids=token_type_ids, position_ids=position_ids, past_key_values_length=past_key_values_length)
        # step2: local transformer
        # batch_size, len, dim
        embedding = self.local_transformer(embedding, attention_mask=mask)[0]
        # step3: conv downsample
        # batch_size, dim, len
        down_sampled_embedding = self.conv_downsample(embedding.transpose(1, 2))
        down_sampled_embedding = down_sampled_embedding.transpose(1, 2)

        return down_sampled_embedding, embedding

    def _hash_bucket_tensors(self, ids, num_hashes, num_buckets):
        """
        Returns:
          A sequence of tensors, each of which is the hash bucket IDs from one hash function.
        """
        if num_hashes > len(_PRIMES):
            raise ValueError("num_hashes must be <= {len(_PRIMES)}")
        primes = _PRIMES[:num_hashes]
        result_tensors = []
        for prime in primes:
            hashed = ((ids + 1) * prime) % num_buckets
            result_tensors.append(hashed)
        return result_tensors

    def forward_up(self, sequence_output, embedding_predown):
        # repeat the sequence for upsample
        # sequence_output: batch_size, len, dim
        up_sample = torch.repeat_interleave(sequence_output, self.config.down_sample_rate, dim=1)
        x = torch.cat([up_sample, embedding_predown], dim=-1)
        # pad to make the output the same size.
        # pad from the last dimension, pad the second dimension which is len.
        x = torch.nn.functional.pad(x, (0,0,1,2))
        out = self.conv_upsample(x.transpose(1, 2)).transpose(1, 2)
        out = self.transformer(out)[0]
        return out


    @property
    def weight(self):
        return None

    def char_embedding_postprocess(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        assert inputs_embeds is not None

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


