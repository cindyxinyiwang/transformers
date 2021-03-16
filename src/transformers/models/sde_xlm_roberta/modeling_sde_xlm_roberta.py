# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch XLM-RoBERTa model. """

from ...activations import ACT2FN, gelu
from ...file_utils import add_start_docstrings
from ...utils import logging
from ..roberta.modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
    RobertaPreTrainedModel,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .configuration_sde_xlm_roberta import SDEXLMRobertaConfig

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss


logger = logging.get_logger(__name__)

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "xlm-roberta-large-finetuned-conll02-dutch",
    "xlm-roberta-large-finetuned-conll02-spanish",
    "xlm-roberta-large-finetuned-conll03-english",
    "xlm-roberta-large-finetuned-conll03-german",
    # See all XLM-RoBERTa models at https://huggingface.co/models?filter=xlm-roberta
]


XLM_ROBERTA_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.XLMRobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
class SDEXLMRobertaModel(RobertaModel):
    """
    This class overrides :class:`~transformers.RobertaModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = SDEXLMRobertaConfig


@add_start_docstrings(
    "XLM-RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.",
    XLM_ROBERTA_START_DOCSTRING,
)
class SDEXLMRobertaForCausalLM(RobertaForCausalLM):
    """
    This class overrides :class:`~transformers.RobertaForCausalLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = SDEXLMRobertaConfig


@add_start_docstrings(
    """XLM-RoBERTa Model with a `language modeling` head on top. """,
    XLM_ROBERTA_START_DOCSTRING,
)

class SDEXLMRobertaForMaskedLM(RobertaPreTrainedModel):
    """
    This class overrides :class:`~transformers.RobertaForMaskedLM`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = SDEXLMRobertaConfig
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = SDERobertaLMHead(config)

        self.init_weights()
        self.sde_embed = config.sde_embed
        self.config = config

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        indices_replaced=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        max_len = indices_replaced.size(1)
        embeds = self.roberta.get_input_embeddings()(input_ids, max_len)
        inputs_embeds = embeds.clone()
        inputs_embeds[indices_replaced, :] = self.roberta.get_input_embeddings().mask_embedding()
        outputs = self.roberta(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        # bsize, len, NCE_vsize
        prediction_scores = self.lm_head(sequence_output, sde_embeds=embeds)
        bsize, max_len, nce_vsize = prediction_scores.shape
        masked_lm_loss = None
        labels = torch.arange(nce_vsize).view(bsize, max_len).to(prediction_scores.device)
        labels[~indices_replaced] = -100
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, nce_vsize), labels.view(-1))
        if self.config.sde_selfnorm_w > 0:
            selfnorm_loss = torch.nn.functional.log_softmax(prediction_scores, dim=-1)
            selfnorm_loss = selfnorm_loss**2 / (bsize*max_len)
            masked_lm_loss = masked_lm_loss + self.config.sde_selfnorm_w * selfnorm_loss.sum()
        #print(masked_lm_loss)
        #print(labels)
        #exit(0)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SDERobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        ## Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        #self.decoder.bias = self.bias
        self.sde_embed = config.sde_embed

    def forward(self, features, sde_embeds, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        #if self.sde_embed and self.sde_embed != "combine":
        #    sde_embed = kwargs.get("sde_embed")
        #    #x = torch.nn.functional.linear(x, sde_embed, bias=self.bias)
        #    x = torch.nn.functional.linear(x, sde_embed)
        #else:
        #    x = self.decoder(x)

        # sde_embed should be [NCE_vsize, hidden_size]
        x = torch.nn.functional.linear(x, sde_embeds.view(-1, sde_embeds.size(2)))
        return x



@add_start_docstrings(
    """
    XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class SDEXLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = SDEXLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class SDEXLMRobertaForMultipleChoice(RobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.RobertaForMultipleChoice`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = SDEXLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class SDEXLMRobertaForTokenClassification(RobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.RobertaForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = SDEXLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class SDEXLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.RobertaForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = SDEXLMRobertaConfig
