import torch
import torch.nn as nn


class CharacterNgramEmbedder(torch.nn.Module):
    def __init__(
            self,
            vocab_size,
            char_embed_dim,
            word_embed_dim,
            pad_token_id,
            highway_layers=0,
            char_emb=None,
    ):
        super(CharacterNgramEmbedder, self).__init__()

        self.embedding_dim = word_embed_dim
        if char_emb is not None:
            self.char_embeddings = char_emb
        else:
            self.char_embeddings = nn.Embedding(vocab_size, char_embed_dim, padding_idx=vocab.pad())
        self.pad_token_id = pad_token_id 

        last_dim = char_embed_dim

        self.highway = Highway(last_dim, highway_layers) if highway_layers > 0 else None

        if char_embed_dim != word_embed_dim:
            self.projection = nn.Linear(last_dim, word_embed_dim)
        else:
            self.projection = None

        #self.reset_parameters()

    def reset_parameters(self):
        #nn.init.xavier_normal_(self.char_embeddings.weight)
        #nn.init.xavier_normal_(self.symbol_embeddings)

        nn.init.constant_(self.char_embeddings.weight[self.char_embeddings.padding_idx], 0.)
        if self.projection is not None:
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.constant_(self.projection.bias, 0.)

    def forward(
            self,
            input,
    ):
        chars = input.view(-1, input.size(-1)) 
        word_embs = self._ave(chars)
        return word_embs.view(input.size()[:2] + (-1,))

    def _ave(
            self,
            char_idxs: torch.Tensor,
    ):
        # BTC
        char_embs = self.char_embeddings(char_idxs)
        pad_mask = char_idxs.eq(self.pad_token_id)
        char_embs[pad_mask] = 0.
        # BC
        x = char_embs.sum(dim=1)
        non_pad_mask = 1-pad_mask.int()
        char_counts = torch.clamp(non_pad_mask.sum(dim=-1), min=1)
        x = x/char_counts.unsqueeze(-1)

        if self.highway is not None:
            x = self.highway(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class SDE(nn.Module):
    def __init__(self, vocab_size, dim, pad_token_id, cls_token_id, sep_token_id, unk_token_id, char_emb=None, latent=5000):
        super(SDE, self).__init__()
        # dict, char_dim, word_dim, padding_idx
        self.char_ngram_embedder = CharacterNgramEmbedder(vocab_size, dim, dim, pad_token_id=pad_token_id, highway_layers=0, char_emb=char_emb)
        self.pad_token_id = pad_token_id 
        self.unk_token_id = unk_token_id 
        self.sep_token_id = sep_token_id 
        self.cls_token_id = cls_token_id 
        self.embedding_dim = dim
        self.preset_emb = char_emb

        
        if latent > 0:
            self.latent_mat = nn.Parameter(torch.empty(latent, dim))
            # run weight initialization
            nn.init.normal_(self.latent_mat, mean=0, std=latent ** -0.5)
        else:
            self.latent_mat = None

    def forward(self, x):
        # build current iteration weight matrix
        # BOW
        ngram_weight = self.char_ngram_embedder(x)  # N_ng * dim
        ngram_weight = torch.tanh(ngram_weight)

        # lang specific
        lang_emb = ngram_weight
        #lang_emb = self.language_transformations[lang_pair](ngram_weight)  # N_ng * dim
        #lang_emb = torch.tanh(lang_emb)
        # latent
        if self.latent_mat is not None:
            latent_scores = torch.matmul(lang_emb, self.latent_mat.transpose(0, 1))
            latent_distribution = torch.softmax(latent_scores, dim=-1)
            latent_emb = torch.matmul(latent_distribution, self.latent_mat)
        # residual connection
            sde_emb = lang_emb + latent_emb  # threshold * dim
        else:
            sde_emb = lang_emb

        if self.preset_emb is not None:
            batch_size, x_len, max_char_len = x.size()
            chars = x.view(-1, max_char_len)
            sde_emb = sde_emb.view(batch_size*x_len, -1)

            pads = chars[:, 0].eq(self.pad_token_id)
            unks = chars[:, 0].eq(self.unk_token_id)
            seps = chars[:, 0].eq(self.sep_token_id)
            clss = chars[:, 0].eq(self.cls_token_id)
            
            if pads.any():
                sde_emb[pads] = self.preset_emb.weight[self.pad_token_id]
            if unks.any():
                sde_emb[unks] = self.preset_emb.weight[self.unk_token_id]
            if seps.any():
                sde_emb[seps] = self.preset_emb.weight[self.sep_token_id]
            if clss.any():
                sde_emb[clss] = self.preset_emb.weight[self.cls_token_id]

            sde_emb = sde_emb.view(batch_size, x_len, -1)
        return sde_emb


class precalcSDE(nn.Module):
    def __init__(self, tokenizer, config, pairs=None, ngram_pool_mode='mean', n=4, threshold=32000, dim=128, latent=10000,
                 do_layer_norm=False):
        super(precalcSDE, self).__init__()
        self.padding_idx = tokenizer.pad_token_id
        dictionary = list(tokenizer.get_vocab().keys())
        self.embedding_dim = dim
        word_vocab = dictionary[4:-1]
        print(dictionary[:5])
        print(dictionary[-1])
        word_ngrams = [self.to_ngram(w, n=n) for w in word_vocab]

        from collections import Counter
        all_ngrams = Counter(sum(word_ngrams, []))
        top_ngrams = all_ngrams.most_common(threshold)
        print(f'BUILDING SDE LAYER, using n={n}, TOTAL ngram {len(all_ngrams)}, suppressing to {len(top_ngrams)}')
        print(f'ngram cutoff at {top_ngrams[-1][1]}')
        top_ngrams_symbols = [x[0] for x in top_ngrams]
        ngram_to_id = {s: i + 1 for i, s in enumerate(top_ngrams_symbols)}
        UNK_ID = 0
        ngram_to_id['<UNK>'] = UNK_ID

        #ngrams_id = [[ngram_to_id.get(w, UNK_ID) for w in ww] for ww in word_ngrams]
        ngrams_id = []
        for ww in word_ngrams:
            ids = []
            for w in ww:
                if w in ngram_to_id: ids.append(ngram_to_id[w])
            ngrams_id.append(ids)
        ngram_offsets = [0]
        for xx in ngrams_id[1:]:
            ngram_offsets.append(ngram_offsets[-1] + len(xx))
        self.register_buffer('ngram_offsets', torch.LongTensor(ngram_offsets))
        self.register_buffer('ngram_ids', torch.LongTensor(sum(ngrams_id, [])))

        self.dim = dim
        self.ngram_emb = nn.EmbeddingBag(len(ngram_to_id), embedding_dim=dim, mode=ngram_pool_mode)
        #self.language_transformations = nn.ModuleDict({
        #    p: nn.Linear(dim, dim, bias=False) for p in pairs
        #})
        if latent > 0:
            self.latent_mat = nn.Parameter(torch.empty(latent, dim))
        else:
            self.latent_mat = None
        self.special_emb = nn.Parameter(torch.empty(4, dim))
        self.mask_emb = nn.Parameter(torch.empty(1, dim))

        if do_layer_norm:
            self.layer_norm = LayerNorm(dim)
        else:
            self.layer_norm = None

        # build current iteration weight matrix
        # BOW
        ngram_weight = self.ngram_emb(self.ngram_ids, self.ngram_offsets)  # N_ng * dim
        ngram_weight = torch.tanh(ngram_weight)
        # lang specific
        lang_emb = ngram_weight
        #lang_emb = self.language_transformations[lang_pair](ngram_weight)  # N_ng * dim
        #lang_emb = torch.tanh(lang_emb)
        # latent
        if self.latent_mat is not None:
            latent_scores = torch.matmul(lang_emb, self.latent_mat.transpose(0, 1))
            latent_distribution = torch.softmax(latent_scores, dim=-1)
            latent_emb = torch.matmul(latent_distribution, self.latent_mat)
            # residual connection
            token_emb = lang_emb + latent_emb  # threshold * dim
        else:
            token_emb = lang_emb

        emb_weight = torch.cat((self.special_emb, token_emb, self.mask_emb), dim=0)
        self.sde_weight = emb_weight
        self.config = config

    def init_weight(self):
        # run weight initialization
        #nn.init.normal_(self.ngram_emb.weight, mean=0, std=self.dim ** -0.5)
        #if self.latent_mat is not None:
        #    nn.init.normal_(self.latent_mat, mean=0, std=self.dim ** -0.5)
        #nn.init.normal_(self.special_emb, mean=0, std=self.dim ** -0.5)
        #nn.init.normal_(self.mask_emb, mean=0, std=self.dim ** -0.5)
        #nn.init.constant_(self.special_emb[self.padding_idx], 0.0)

        nn.init.normal_(self.ngram_emb.weight, mean=0, std=self.config.initializer_range*0.5)
        if self.latent_mat is not None:
            nn.init.normal_(self.latent_mat, mean=0, std=self.config.initializer_range*0.5)
        nn.init.normal_(self.special_emb, mean=0, std=self.config.initializer_range*0.5)
        nn.init.normal_(self.mask_emb, mean=0, std=self.config.initializer_range*0.5)
        nn.init.constant_(self.special_emb[self.padding_idx], 0.0)



    @staticmethod
    def to_ngram(word: str, n=4):
        ngrams = []
        for l in range(1, min(n + 1, len(word) + 1)):
            for start, end in zip(range(len(word)), range(l, len(word) + 1)):
                ngrams.append(word[start:end])
        return ngrams

    def forward(self, x, lang_pair=None):
        # build current iteration weight matrix
        # BOW
        ngram_weight = self.ngram_emb(self.ngram_ids, self.ngram_offsets)  # N_ng * dim
        ngram_weight = torch.tanh(ngram_weight)
        # lang specific
        lang_emb = ngram_weight
        #lang_emb = self.language_transformations[lang_pair](ngram_weight)  # N_ng * dim
        #lang_emb = torch.tanh(lang_emb)
        # latent
        if self.latent_mat is not None:
            latent_scores = torch.matmul(lang_emb, self.latent_mat.transpose(0, 1))
            latent_distribution = torch.softmax(latent_scores, dim=-1)
            latent_emb = torch.matmul(latent_distribution, self.latent_mat)
            # residual connection
            token_emb = lang_emb + latent_emb  # threshold * dim
        else:
            token_emb = lang_emb
        emb_weight = torch.cat((self.special_emb, token_emb, self.mask_emb), dim=0)
        self.sde_weight = emb_weight
        sde_emb = nn.functional.embedding(x, emb_weight, padding_idx=self.padding_idx)
        if self.layer_norm:
            sde_emb = self.layer_norm(sde_emb)
        return sde_emb

    @property
    def weight(self):
        #return self.sde_weight
        return torch.nn.Parameter(self.sde_weight)


class SDENoWeight(precalcSDE):
    def forward(self, x, lang):
        # construct mask
        special_mask = (x < len(self.special_emb)).unsqueeze(-1)  # B * L
        non_special_mask = ~special_mask  # B * L
        # build current iteration weight matrix
        # BOW
        ngram_weight = self.ngram_emb(self.ngram_ids, self.ngram_offsets) # N_ng * dim
        ngram_weight = torch.tanh(ngram_weight)
        ngram_emb_weight = torch.cat((self.special_emb, ngram_weight), dim=0) # len(dict) * dim
        lang_indep_emb = nn.functional.embedding(x, ngram_emb_weight, padding_idx=self.padding_idx)  # B * L * dim
        # lang specific
        lang_emb = self.language_transformations[lang](lang_indep_emb)  # B * L * dim
        lang_emb = torch.tanh(lang_emb)
        # latent
        latent_scores = torch.matmul(lang_emb, self.latent_mat.transpose(0, 1))
        latent_distribution = torch.softmax(latent_scores, dim=-1)
        latent_emb = torch.matmul(latent_distribution, self.latent_mat)
        # residual connection
        final_word_emb = lang_emb + latent_emb
        # piece up things
        final_word_emb = non_special_mask.type(final_word_emb.dtype) * final_word_emb
        special_emb = special_mask.type(lang_indep_emb.dtype) * lang_indep_emb
        final_emb = final_word_emb + special_emb
        return final_emb

    @property
    def weight(self):
        raise NotImplementedError(f'{type(self).__name__} does not support getting full weight')



class SDEFull(nn.Module):
    def __init__(self, tokenizer, config, pairs=None, ngram_pool_mode='mean', n=4, threshold=32000, dim=128, latent=10000,
                 do_layer_norm=False):
        super(SDEFull, self).__init__()
        self.padding_idx = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        dictionary = list(tokenizer.get_vocab().keys())
        self.vsize = len(dictionary)
        self.embedding_dim = dim

        self.ngram_proj = nn.Linear(self.vsize, dim, bias=False)
        #self.language_transformations = nn.ModuleDict({
        #    p: nn.Linear(dim, dim, bias=False) for p in pairs
        #})
        if latent > 0:
            self.latent_mat = nn.Parameter(torch.empty(latent, dim))
        else:
            self.latent_mat = None
        #self.special_emb = nn.Parameter(torch.empty(4, dim))
        #self.mask_emb = nn.Parameter(torch.empty(1, dim))

        if do_layer_norm:
            self.layer_norm = LayerNorm(dim)
        else:
            self.layer_norm = None
        self.config = config

    def mask_embedding(self):
        return self.ngram_proj.weight[:,self.tokenizer.mask_token_id]

    def init_weight(self):
        nn.init.normal_(self.ngram_proj.weight, mean=0, std=self.config.initializer_range*0.5)
        if self.latent_mat is not None:
            nn.init.normal_(self.latent_mat, mean=0, std=self.config.initializer_range*0.5)
        nn.init.constant_(self.ngram_proj.weight[self.padding_idx], 0.0)
        #nn.init.normal_(self.special_emb, mean=0, std=self.config.initializer_range*0.5)
        #nn.init.normal_(self.mask_emb, mean=0, std=self.config.initializer_range*0.5)

    def forward(self, x, max_len, lang_pair=None):
        # x: [batch_size, [coo, vals]]
        # BOW
        if type(x) == list:
            bow_embs = []
            for sparse_data in x:
                emb = torch.sparse_coo_tensor(sparse_data[0], sparse_data[1], (max_len, self.vsize), device=self.ngram_proj.weight.device, dtype=self.ngram_proj.weight.dtype)
                bow_embs.append(emb)
            bow_embs = torch.stack(bow_embs, dim=0)
            #ngram_weight = torch.tanh(self.ngram_proj(bow_embs.to_dense().float()))
            ngram_weight = torch.tanh(self.ngram_proj(bow_embs.to_dense()))
            #bow_embs = []
            #for sparse_data in x:
            #    emb = torch.sparse_coo_tensor(sparse_data[0], sparse_data[1], (max_len, self.vsize)).to_dense().to(self.ngram_proj.weight.device)
            #    emb = torch.tanh(self.ngram_proj(emb.float()))
            #    #emb = torch.sparse_coo_tensor(sparse_data[0], sparse_data[1], (max_len, self.vsize)).to_dense().to(self.ngram_proj.weight.device).float()
            #    #if self.config.sde_ave:
            #    #    emb = emb / emb.sum(dim=-1, keepdim=True) 
            #    #emb = torch.tanh(self.ngram_proj(emb))
            #    bow_embs.append(emb)
            #ngram_weight = torch.stack(bow_embs, dim=0)
        else:
            ngram_weight = torch.tanh(self.ngram_proj(x.to_dense().float()))
            #emb = x.to_dense().float()
            #if self.config.sde_ave:
            #    emb = emb / emb.sum(dim=-1, keepdim=True) 
            #ngram_weight = torch.tanh(self.ngram_proj(emb))

        # lang specific
        lang_emb = ngram_weight
        #lang_emb = self.language_transformations[lang_pair](ngram_weight)  # N_ng * dim
        #lang_emb = torch.tanh(lang_emb)
        # latent
        if self.latent_mat is not None:
            latent_scores = torch.matmul(lang_emb, self.latent_mat.transpose(0, 1))
            latent_distribution = torch.softmax(latent_scores, dim=-1)
            latent_emb = torch.matmul(latent_distribution, self.latent_mat)
            # residual connection
            token_emb = lang_emb + latent_emb  # threshold * dim
        else:
            token_emb = lang_emb
        if self.layer_norm:
            sde_emb = self.layer_norm(sde_emb)
        return token_emb 

    @property
    def weight(self):
        return None

