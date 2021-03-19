# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
# limitations under the License
""" Tokenization classes for XLM-RoBERTa model."""


import os
from shutil import copyfile
from typing import List, Optional, Tuple, Any, Dict, Union
#from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import sentencepiece as spm

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TensorType


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "char4gram32k.vocab"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model",
        "xlm-roberta-large": "https://huggingface.co/xlm-roberta-large/resolve/main/sentencepiece.bpe.model",
        "xlm-roberta-large-finetuned-conll02-dutch": "https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/sentencepiece.bpe.model",
        "xlm-roberta-large-finetuned-conll02-spanish": "https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/sentencepiece.bpe.model",
        "xlm-roberta-large-finetuned-conll03-english": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/sentencepiece.bpe.model",
        "xlm-roberta-large-finetuned-conll03-german": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-german/resolve/main/sentencepiece.bpe.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlm-roberta-base": 512,
    "xlm-roberta-large": 512,
    "xlm-roberta-large-finetuned-conll02-dutch": 512,
    "xlm-roberta-large-finetuned-conll02-spanish": 512,
    "xlm-roberta-large-finetuned-conll03-english": 512,
    "xlm-roberta-large-finetuned-conll03-german": 512,
}


class SDEXLMRobertaTokenizer(PreTrainedTokenizer):
    """
    Adapted from :class:`~transformers.RobertaTokenizer` and class:`~transformers.XLNetTokenizer`. Based on
    `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.

    Attributes:
        sp_model (:obj:`SentencePieceProcessor`):
            The `SentencePiece` processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        n=4,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self._load_vocab(self.vocab_file)
        self.n = n
        # keys: char_ngram ids, vals: count for each ids
        self.bos_token_charkv = [[self._charngram_to_id(bos_token)], [1]] 
        self.eos_token_charkv = [[self._charngram_to_id(eos_token)], [1]]
        self.sep_token_charkv = [[self._charngram_to_id(sep_token)], [1]]
        self.cls_token_charkv = [[self._charngram_to_id(cls_token)], [1]]
        self.unk_token_charkv = [[self._charngram_to_id(unk_token)], [1]]
        self.pad_token_charkv = [[self._charngram_to_id(pad_token)], [1]]
        self.mask_token_charkv = [[self._charngram_to_id(mask_token)], [1]]


    def _load_vocab(self, f):
        # construct tokens_to_ids and ids_to_tokens
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self.fairseq_ids_to_tokens = {}
        with open(f, 'r') as word_file:
            for line in word_file:
                tokens = line.strip().split("\t")
                if len(tokens) != 2:
                    logger.info("The line from vocab file might be wrong {}".format(line))
                w = tokens[0]
                self.fairseq_tokens_to_ids[w] = len(self.fairseq_tokens_to_ids)
        self.fairseq_tokens_to_ids["<mask>"] = len(self.fairseq_tokens_to_ids)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self._load_vocab(self.vocab_file)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[Dict], token_ids_1: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM-RoBERTa sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_charkv] + token_ids_0 + [self.sep_token_charkv]
        cls = [self.cls_token_charkv]
        sep = [self.sep_token_charkv]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[Dict], token_ids_1: Optional[List[Dict]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
        not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of zeros.

        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        #return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token
        return len(self.fairseq_tokens_to_ids)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return text.split()

    def _charngram_to_id(self, char_ngram):
        if char_ngram in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[char_ngram]
        else:
            return self.unk_token_id

    def _token_to_ngram(self, token):
        """ return: key: charngram id; value: count of this ngram """
        if token == self.bos_token: return self.bos_token_charkv
        if token == self.eos_token: return self.eos_token_charkv
        if token == self.sep_token: return self.sep_token_charkv
        if token == self.cls_token: return self.cls_token_charkv
        if token == self.unk_token: return self.unk_token_charkv
        if token == self.pad_token: return self.pad_token_charkv
        if token == self.mask_token: return self.mask_token_charkv

        char_kv = {}
        for i in range(len(token)):
            for j in range(i+1, min(i+self.n, len(token))+1):
                char_ngram = token[i:j]
                char_id = self._charngram_to_id(char_ngram)
                #char_kv[char_id] = char_kv.get(char_id, 0) + 1
                if char_id != self.unk_token_id:
                    char_kv[char_id] = char_kv.get(char_id, 0) + 1
        #print(char_kv)
        #exit(0)
        # convert to k, v list
        char_kv = [list(char_kv.keys()), list(char_kv.values())]
        return char_kv

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self._token_to_ngram(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        else:
            logger.info("Warning: index {} out of range".format(index))

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = " ".join(tokens)
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns :obj:`None` if the token
        has not been set.
        """
        if self._bos_token is None:
            return None
        return self.fairseq_tokens_to_ids[self.bos_token]

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the end of sentence token in the vocabulary. Returns :obj:`None` if the token has
        not been set.
        """
        if self._eos_token is None:
            return None
        return self.fairseq_tokens_to_ids[self.eos_token]

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the unknown token in the vocabulary. Returns :obj:`None` if the token has not been
        set.
        """
        if self._unk_token is None:
            return None
        return self.fairseq_tokens_to_ids[self.unk_token]

    @property
    def sep_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
        sequence. Returns :obj:`None` if the token has not been set.
        """
        if self._sep_token is None:
            return None
        return self.fairseq_tokens_to_ids[self.sep_token]

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the padding token in the vocabulary. Returns :obj:`None` if the token has not been
        set.
        """
        if self._pad_token is None:
            return None
        return self.fairseq_tokens_to_ids[self.pad_token]

    @property
    def pad_token_type_id(self) -> int:
        """
        :obj:`int`: Id of the padding token type in the vocabulary.
        """
        return self._pad_token_type_id

    @property
    def cls_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input
        sequence leveraging self-attention along the full depth of the model.

        Returns :obj:`None` if the token has not been set.
        """
        if self._cls_token is None:
            return None
        return self.fairseq_tokens_to_ids[self.cls_token]

    @property
    def mask_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
        modeling. Returns :obj:`None` if the token has not been set.
        """
        if self._mask_token is None:
            return None
        return self.fairseq_tokens_to_ids[self.mask_token]

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, List[int]],
            Dict[str, List[List[int]]],
            List[Dict[str, List[int]]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with ``self.padding_side``,
        ``self.pad_token_id`` and ``self.pad_token_type_id``)

        .. note::

            If the ``encoded_inputs`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
            result will use the same type unless you provide a different tensor type with ``return_tensors``. In the
            case of PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            encoded_inputs (:class:`~transformers.BatchEncoding`, list of :class:`~transformers.BatchEncoding`, :obj:`Dict[str, List[int]]`, :obj:`Dict[str, List[List[int]]` or :obj:`List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input (:class:`~transformers.BatchEncoding` or :obj:`Dict[str,
                List[int]]`) or a batch of tokenized inputs (list of :class:`~transformers.BatchEncoding`, `Dict[str,
                List[List[int]]]` or `List[Dict[str, List[int]]]`) so you can use this method during preprocessing as
                well as in a PyTorch Dataloader collate function.

                Instead of :obj:`List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], (dict, BatchEncoding)):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method"
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        if isinstance(first_element, dict):
            first_element = list(first_element.keys())[0]
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)
        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors, char_vsize=self.vocab_size)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        return BatchEncoding(batch_outputs, tensor_type=return_tensors, char_vsize=self.vocab_size)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, List[int]], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(required_input) + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_charkv] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(required_input)
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_charkv] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)
 
        return encoded_inputs

