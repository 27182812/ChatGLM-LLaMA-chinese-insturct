# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Whisper."""
import json
import os
from typing import List, Optional, Tuple

import numpy as np
from tokenizers import pre_tokenizers, processors

from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .english_normalizer import EnglishTextNormalizer
from .tokenization_whisper import LANGUAGES, TASK_IDS, TO_LANGUAGE_CODE, WhisperTokenizer, _decode_asr


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_file": "tokenizer.json",
    "merges_file": "merges.txt",
    "normalizer_file": "normalizer.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/vocab.json",
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/vocab.json",
        "openai/whisper-small": "https://huggingface.co/openai/whisper-small/resolve/main/vocab.json",
        "openai/whisper-medium": "https://huggingface.co/openai/whisper-medium/resolve/main/vocab.json",
        "openai/whisper-large": "https://huggingface.co/openai/whisper-large/resolve/main/vocab.json",
        "openai/whisper-tiny.en": "https://huggingface.co/openai/whisper-tiny.en/resolve/main/vocab.json",
        "openai/whisper-base.en": "https://huggingface.co/openai/whisper-base.en/resolve/main/vocab.json",
        "openai/whisper-small.en": "https://huggingface.co/openai/whisper-small.en/resolve/main/vocab.json",
        "openai/whisper-medium.en": "https://huggingface.co/openai/whisper-medium.en/resolve/main/vocab.json",
    },
    "merges_file": {
        "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/merges.txt",
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/merges.txt",
        "openai/whisper-small": "https://huggingface.co/openai/whisper-small/resolve/main/merges.txt",
        "openai/whisper-medium": "https://huggingface.co/openai/whisper-medium/resolve/main/merges.txt",
        "openai/whisper-large": "https://huggingface.co/openai/whisper-large/resolve/main/merges.txt",
        "openai/whisper-tiny.en": "https://huggingface.co/openai/whisper-tiny.en/resolve/main/merges.txt",
        "openai/whisper-base.en": "https://huggingface.co/openai/whisper-base.en/resolve/main/merges.txt",
        "openai/whisper-small.en": "https://huggingface.co/openai/whisper-small.en/resolve/main/merges.txt",
        "openai/whisper-medium.en": "https://huggingface.co/openai/whisper-medium.en/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json",
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/tokenizer.json",
        "openai/whisper-small": "https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json",
        "openai/whisper-medium": "https://huggingface.co/openai/whisper-medium/resolve/main/tokenizer.json",
        "openai/whisper-large": "https://huggingface.co/openai/whisper-large/resolve/main/tokenizer.json",
        "openai/whisper-tiny.en": "https://huggingface.co/openai/whisper-tiny.en/resolve/main/tokenizer.json",
        "openai/whisper-base.en": "https://huggingface.co/openai/whisper-base.en/resolve/main/tokenizer.json",
        "openai/whisper-small.en": "https://huggingface.co/openai/whisper-small.en/resolve/main/tokenizer.json",
        "openai/whisper-medium.en": "https://huggingface.co/openai/whisper-medium.en/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai/whisper-tiny": 1500,
    "openai/whisper-base": 1500,
    "openai/whisper-small": 1500,
    "openai/whisper-medium": 1500,
    "openai/whisper-large": 1500,
    "openai/whisper-tiny.en": 1500,
    "openai/whisper-base.en": 1500,
    "openai/whisper-small.en": 1500,
    "openai/whisper-medium.en": 1500,
}


class WhisperTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Whisper tokenizer (backed by HuggingFace's *tokenizers* library).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        normalizer_file (`str`, *optional*, defaults to `None`):
            Path to the normalizer_file file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `<|startoftranscript|>`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Whisper tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether or not the post-processing step should trim offsets to avoid including whitespaces.
        language (`str`, *optional*):
            The language of the transcription text. The corresponding language id token is appended to the start of the
            sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
            `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
        task (`str`, *optional*):
            Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
            fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
        predict_timestamps (`bool`, *optional*, defaults to `False`):
            Whether to omit the `<|notimestamps|>` token at the start of the sequence.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = WhisperTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        normalizer_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|startoftranscript|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        language=None,
        task=None,
        predict_timestamps=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.add_bos_token = kwargs.pop("add_bos_token", False)

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        if normalizer_file is not None:
            with open(normalizer_file, encoding="utf-8") as vocab_handle:
                self.english_spelling_normalizer = json.load(vocab_handle)
        else:
            self.english_spelling_normalizer = None

        self.add_prefix_space = add_prefix_space

        self.language = language
        self.task = task
        self.predict_timestamps = predict_timestamps

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast._batch_encode_plus
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast._encode_plus
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._decode_with_timestamps
    def _decode_with_timestamps(self, token_ids, time_precision=0.02) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        timestamp_begin = self.all_special_ids[-1] + 1
        outputs = [[]]
        for token in token_ids:
            if token >= timestamp_begin:
                timestamp = f"<|{(token - timestamp_begin) * time_precision:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [s if isinstance(s, str) else self.decode(s) for s in outputs]
        return "".join(outputs)

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._compute_offsets
    def _compute_offsets(self, token_ids, time_precision=0.02):
        """
        Compute offsets for a given tokenized input

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        offsets = []
        token_ids = np.array(token_ids)
        if token_ids.shape[0] > 1 and len(token_ids.shape) > 1:
            raise ValueError("Can only process a single input at a time")
        timestamp_begin = self.all_special_ids[-1] + 1
        timestamp_tokens = token_ids >= timestamp_begin

        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        if consecutive.shape[0] == 0 and timestamp_tokens.sum() <= 1:
            # either there are no timestamps or there are no consecutive ones
            return []
        elif np.where(timestamp_tokens)[0][-1] + 1 not in consecutive:
            # we add the final timestamp if it is not already in the list
            consecutive = np.append(consecutive, np.where(timestamp_tokens)[0][-1] + 1)

        last_slice = np.where(timestamp_tokens)[0][0]
        for current_slice in consecutive:
            sliced_tokens = token_ids[last_slice:current_slice]
            if len(sliced_tokens) > 1:
                start_timestamp_position = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_position = sliced_tokens[-1].item() - timestamp_begin
                offsets.append(
                    {
                        "text": self._decode(sliced_tokens),
                        "timestamp": (
                            start_timestamp_position * time_precision,
                            end_timestamp_position * time_precision,
                        ),
                    }
                )
            last_slice = current_slice

        return offsets

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.decode
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        output_offsets: bool = False,
        time_precision=0.02,
        decode_with_timestamps: bool = False,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.
            output_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output the offsets of the tokens. This should only be set if the model predicted
                timestamps.
            decode_with_timestamps (`bool`, *optional*, defaults to `False`):
                WHether or not to decode with timestamps included in the raw text.
        Returns:
            `str`: The decoded sentence.
        """
        text = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        if decode_with_timestamps:
            text = self._decode_with_timestamps(token_ids, time_precision=time_precision)
        # retrieve offsets
        if output_offsets:
            offsets = None
            offsets = self._compute_offsets(token_ids, time_precision=time_precision)
            return {"text": text, "offsets": offsets}
        return text

    def _decode(self, *args, normalize: bool = False, **kwargs) -> str:
        text = super()._decode(*args, **kwargs)

        if normalize:
            clean_text = self._normalize(text)
            return clean_text
        else:
            return text

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._normalize
    def _normalize(self, text):
        """
        Normalize a given string using the `EnglishTextNormalizer` class, which preforms commons transformation on
        english text.
        """
        normalizer = EnglishTextNormalizer(self.english_spelling_normalizer)
        return normalizer(text)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)

        normalizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["normalizer_file"]
        )

        if self.english_spelling_normalizer is not None:
            with open(normalizer_file, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(self.english_spelling_normalizer, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                )

        return tuple(files) + (normalizer_file,)

    def set_prefix_tokens(self, language: str = None, task: str = None, predict_timestamps: bool = None):
        """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning. Example:

        ```python
        >>> # instantiate the tokenizer and set the prefix token to Spanish
        >>> tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny", language="spanish")
        >>> # now switch the prefix token from Spanish to French
        >>> tokenizer.set_prefix_tokens(language="french")
        ```

        Args:
            language (`str`, *optional*, defaults to `None`):
                The language of the transcription text.
            task (`str`, *optional*, defaults to `None`):
                Task identifier to append at the start of sequence (if any).
            predict_timestamps (`bool`, *optional*, defaults to `None`):
                Whether to omit the `<|notimestamps|>` token at the start of the sequence.
        """
        self.language = language if language is not None else self.language
        self.task = task if task is not None else self.task
        self.predict_timestamps = predict_timestamps if predict_timestamps is not None else self.predict_timestamps

        prefix_token_ids = self.prefix_tokens
        prefixes = self.convert_ids_to_tokens(prefix_token_ids)
        eos = self.eos_token
        eos_token_id = self.eos_token_id
        prefix_template = " ".join([f"{token}:0" for token in prefixes])
        self.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{prefix_template} $A:0 {eos}:0",
            pair=f"{prefix_template} $A:0 $B:1 {eos}:1",
            special_tokens=[
                (eos, eos_token_id),
                *zip(prefixes, prefix_token_ids),
            ],
        )

    @property
    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.prefix_tokens
    def prefix_tokens(self) -> List[int]:
        all_special_ids = self.all_special_ids
        bos_token_id = all_special_ids[-106]
        translate_token_id = all_special_ids[-6]
        transcribe_token_id = all_special_ids[-5]
        notimestamps_token_id = all_special_ids[-1]
        langs = tuple(LANGUAGES.keys())

        if self.language is not None:
            self.language = self.language.lower()
            if self.language in TO_LANGUAGE_CODE:
                language_id = TO_LANGUAGE_CODE[self.language]
            elif self.language in TO_LANGUAGE_CODE.values():
                language_id = self.language
            else:
                is_language_code = len(self.language) == 2
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )

        if self.task is not None:
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        bos_sequence = [bos_token_id]
        if self.language is not None:
            bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
        if self.task is not None:
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else translate_token_id)
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1]
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._build_conversation_input_ids
    def _build_conversation_input_ids(self, conversation) -> List[int]:
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.encode(text, add_special_tokens=False) + [self.eos_token_id])
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length :]
        return input_ids

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.get_decoder_prompt_ids
    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)
        # prefix tokens are of the form: <|startoftranscript|> <|lang_id|> <|task|> <|notimestamps|>
        # we don't want to force the bos token at position 1, as this is the starting token
        # when we generate, so we slice the prefix tokens to: <|lang_id|> <|task|> <|notimestamps|>
        # to get the forced tokens
        forced_tokens = self.prefix_tokens[1:]
        forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_tokens)]
        return forced_decoder_ids

    def _decode_asr(self, model_outputs, *, return_timestamps, return_language, time_precision):
        return _decode_asr(
            self,
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )
