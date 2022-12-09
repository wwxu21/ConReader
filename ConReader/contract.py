# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import json
import random
import math
import re
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import multiprocessing
from multiprocessing import Manager
import numpy as np
from tqdm import tqdm

from transformers.file_utils import is_tf_available, is_torch_available
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy
from transformers.utils import logging
from transformers.data.processors.utils import DataProcessor


# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}


if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.get_logger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def contract_convert_example_to_features(
    example, max_seq_length, max_query_length, doc_stride, padding_strategy, reserved, is_training, tokenizer
):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    definition_ids = []
    definition_type_ids = []
    definition_attention_mask = []
    if is_training and len(example.definitions) != 0:
        definitions = example.definitions

        for definition in definitions:
            key_text = definition['key']
            value_start = orig_to_tok_index[definition['value_position'][0]]
            if definition['value_position'][1] < len(example.doc_tokens) - 1:
                value_end = orig_to_tok_index[definition['value_position'][1] + 1] - 1
            else:
                value_end = len(all_doc_tokens) - 1
            value_text = all_doc_tokens[value_start : value_end + 1]
            definition_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
                key_text,
                value_text,
                truncation=TruncationStrategy.ONLY_SECOND.value,
                padding=padding_strategy,
                max_length=max_query_length,
                return_overflowing_tokens=True,
                return_token_type_ids=True,
            )
            definition_ids.append(definition_dict['input_ids'])
            definition_type_ids.append(definition_dict['token_type_ids'])
            definition_attention_mask.append(definition_dict['attention_mask'])

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )

    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair + reserved

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        # reserve space for reserved repr
        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length - reserved,
            return_overflowing_tokens=True,
            return_token_type_ids=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
        )
        # add space for reserved repr (at the end)
        if reserved != 0:
            encoded_dict['input_ids'] = encoded_dict['input_ids'] + [tokenizer.pad_token_id] * reserved
            encoded_dict['token_type_ids'] = encoded_dict['token_type_ids'] + [encoded_dict['token_type_ids'][0]] * reserved
            encoded_dict['attention_mask'] = encoded_dict['attention_mask'] + [1 - encoded_dict['attention_mask'][0]] * reserved

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )# remaining vs max-length limitation

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len
        encoded_dict['definition_ids'] = definition_ids
        encoded_dict['definition_type_ids'] = definition_type_ids
        encoded_dict['definition_attention_mask'] = definition_attention_mask


        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0

        pad_token_indices = np.where([x == tokenizer.pad_token_id for x in span["input_ids"]])
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False
            ss = tok_start_position >= doc_start
            se = tok_start_position <= doc_end
            es = tok_end_position >= doc_start
            ee = tok_end_position <= doc_end
            if not ((ss and se) or (es and ee)):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens
                if ss and se and es and ee:
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                elif ss and se and not (es and ee):
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = doc_end - doc_start + doc_offset
                elif es and ee and not (ss and se):
                    start_position = doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
        definitions_map = []
        if is_training and len(example.definitions) != 0:
            for definition in example.definitions:
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens
                key_position = definition['key_position']
                definition_map = []
                for one_key in key_position:

                    tok_key_start = orig_to_tok_index[one_key[0]]
                    if one_key[1] < len(example.doc_tokens) - 1:
                        tok_key_end = orig_to_tok_index[one_key[1] + 1] - 1
                    else:
                        tok_key_end = len(all_doc_tokens) - 1
                    if tok_key_start >= doc_start and tok_key_end <= doc_end:
                        key_start = tok_key_start - doc_start + doc_offset
                        key_end = tok_key_end - doc_start + doc_offset
                        definition_map.extend(range(key_start, key_end+1))

                definitions_map.append(definition_map)



        features.append(
            ContractFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                span['definition_ids'],
                span['definition_type_ids'],
                span['definition_attention_mask'],
                definitions_map,
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
                label_id=example.label_id,
                contract_id=example.contract_id,
            )
        )
    return features


def contract_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def contract_convert_examples_to_features(
    contracts_examples,
    tokenizer,
    max_seq_length,
    max_query_length,
    doc_stride,
    reserved,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    num_worker=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer_c: contract tokenizer :class:`~transformers.PreTrainedTokenizer`
        tokenizer_q: query tokenizer :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset, if 'tf': returns a tf.data.Dataset
        threads: multiple processing threads.


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = contract_convert_examples_to_features(
            examples=examples,
            tokenizer_c=tokenizer_c,
            tokenizer_q=tokenizer_q,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """
    # Defining helper methods
    all_features = []
    manager = Manager()
    results_dict = manager.dict()
    work_load = math.ceil(len(contracts_examples) / num_worker)
    worker_dict = {}
    # work_load=10
    for i_w in range(num_worker):
        if i_w == num_worker - 1:
            work = contracts_examples[i_w * work_load:]
        else:
            work = contracts_examples[i_w * work_load: (i_w + 1) * work_load]
        worker_dict[i_w] = multiprocessing.Process(target=batch_contracts_convert_examples_to_features, args=(
            i_w, results_dict, work, tokenizer, max_seq_length, max_query_length, doc_stride,
            reserved, is_training, padding_strategy,))
        worker_dict[i_w].start()
    for i_w in range(num_worker):
        worker_dict[i_w].join()

    for i_w in range(num_worker):
        all_features.extend(results_dict[i_w])

    # queries_features = [y[0]  for x in all_features for y in x]
    contracts_features = [y for x in all_features for y in x]
    new_contracts_features = []
    unique_id = 1000000000
    example_index = 0

    for contract_features in tqdm(
            contracts_features, total=len(contracts_features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not contract_features:
            continue
        new_contract_features = []
        for example_feature in contract_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_contract_features.append(example_feature)
            unique_id += 1
        example_index += 1
        new_contracts_features.append(new_contract_features)
    contracts_features = new_contracts_features
    del new_contracts_features
    return contracts_features

def batch_contracts_convert_examples_to_features(
        i_w,
        results_dict,
        contracts_examples,
        tokenizer,
        max_seq_length,
        max_query_length,
        doc_stride,
        reserved,
        is_training,
        padding_strategy="max_length",
):
    batch_features = []
    for  contract_examples in tqdm(contracts_examples, desc="convert contract examples to features at worker {}".format(str(i_w))):
        contract_features = []
        for contract_example in contract_examples:
            example_features = contract_convert_example_to_features(
                contract_example, max_seq_length, max_query_length, doc_stride, padding_strategy, reserved, is_training, tokenizer,
            )
            contract_features.append(example_features)
        batch_features.append(contract_features)
    results_dict[i_w] = batch_features

class ContractProcessor(DataProcessor):
    """
    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and
    version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None


    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("ContractProcessor should be instantiated via ContractV1Processor or ContractV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("ContractProcessor should be instantiated via ContractV1Processor or ContractV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        contract_examples = []
        label_dict = {}
        label_id = 0
        for ie, entry in enumerate(tqdm(input_data)):
            # if ie <148 or ie>= 150:
            #     continue
            examples = []
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                definitions = paragraph["definition"]
                def_ready = False
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    label = re.findall(
                        r'(?<=Highlight the parts \(if any\) of this contract related to ").+(?=" that should be reviewed by a lawyer. )',
                        question_text)
                    assert len(label) == 1
                    label = label[0].strip()
                    if label not in label_dict:
                        label_dict[label] = label_id
                        label_id += 1
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = ContractExample(
                        contract_id=ie,
                        qas_id=qas_id,
                        label_id=label_dict[label],
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        def_ready=def_ready,
                        definitions=definitions,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    definitions = example.definitions
                    def_ready = True
                    examples.append(example)
            contract_examples.append(examples)
        return contract_examples


class ContractV1Processor(ContractProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class ContractV2Processor(ContractProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class ContractExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        contract_id,
        qas_id,
        label_id,
        question_text,
        context_text,
        answer_text,
        def_ready,
        definitions,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.contract_id = contract_id
        self.qas_id = qas_id
        self.label_id = label_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]
        # definition position at token-level
        if not def_ready:
            definitions_token = []
            for definition in definitions:
                definition_token = {}
                key = definition['key']
                key_position = definition['key_position']
                value = definition['value']
                value_position = definition['value_position']
                key_position_token = [(char_to_word_offset[start_char],
                                 char_to_word_offset[min(start_char + len(key) - 1, len(char_to_word_offset) - 1)]) for
                                start_char in key_position]
                value_position_token = (char_to_word_offset[value_position],
                                  char_to_word_offset[min(value_position + len(value) - 1, len(char_to_word_offset) - 1)])

                definition_token['key'] = key
                definition_token['value'] = value
                definition_token['key_position'] = key_position_token
                definition_token['value_position'] = value_position_token
                definitions_token.append(definition_token)
            self.definitions = definitions_token
        else:
            self.definitions = definitions
class ContractFeatures:
    """
    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from
    :class:`~transformers.data.processors.squad.SquadExample` using the
    :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
        encoding: optionally store the BatchEncoding with the fast-tokenizer alignement methods.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        definition_ids,
        definition_type_ids,
        definition_attention_mask,
        definition_map,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        qas_id: str = None,
        label_id: int = None,
        contract_id: int = None,
        encoding: BatchEncoding = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.definition_ids = definition_ids
        self.definition_type_ids = definition_type_ids
        self.definition_attention_mask = definition_attention_mask
        self.definition_map = definition_map
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id
        self.label_id = label_id
        self.contract_id = contract_id
        self.encoding = encoding


class ContractResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


class Dataloader(object):
    def __init__(self, args, features, tokenizer, evaluate=False):
        self.tokenizer = tokenizer
        if evaluate:
            self.batch_size = args.per_gpu_eval_batch_size * min(args.n_gpu, 1)
        else:
            self.batch_size = args.per_gpu_train_batch_size * min(args.n_gpu, 1)
        self.is_training = not evaluate
        # Convert to tensor and build dataset
        # query
        self.features = self.get_contract_dataset(features)        # due to the limitation of gpu size, if gpu memory is enough then self.features = features
        self.prepare_batch()
        self.query_size = args.query_size
        self.max_doc_length = args.max_seq_length
        self.max_definition_length = args.max_query_length
        self.reserved = args.reserved
        self.CD = args.CD
    def get_contract_dataset(self, features):
        span = 3
        processed_features = []
        for i_c, c in enumerate(features):
            if len(c) != 0:
                step = 0
                while len(c) - span * step > span:
                    processed_features.append(c[step * span:step * span + span])
                    step += 1
                processed_features.append(c[step * span:])
        return processed_features
    def prepare_batch(self):
        processed_data = self.features
        i = 0
        for i_x, x in enumerate(processed_data):
            for i_y, y in enumerate(x):
                y.eval_id = i
                i += 1
        if self.is_training:
            processed_data = sorted(processed_data, key=lambda x: len(x))
        batches = []
        num, data = 0, []

        for x in processed_data:
            num += len(x)
            data.append(x)
            if num >= self.batch_size:
                batches.append(data)
                num, data = 0, []
        if len(data) != 0:
            batches.append(data)
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        batches = self.batches
        if self.is_training:
            random.shuffle(batches)
        for batch in batches:
            # queries_features = [x[0] for x in batch]
            contracts_features = batch
            max_doc_length = self.max_doc_length
            max_doc_num = max([len(x) for x in contracts_features])
            #definition
            definition_ids = [[self.tokenizer.pad_token_id] * self.max_definition_length]
            definition_type_ids = [[0] * self.max_definition_length]
            definition_attention_mask = [[0] * self.max_definition_length]
            contract_definition_map = []
            definition_exist = {}
            total_lens = 1
            for contract_features in contracts_features:
                contract_definition_map_one = []
                for contract_feature in contract_features:
                    if self.CD:
                        lens = min(len(contract_feature.definition_ids), self.reserved)
                    else:
                        lens = len(contract_feature.definition_ids)
                    contract_definition_map_segment = []
                    for i in range(lens):
                        if contract_feature.definition_map[i] != []:
                            if (contract_feature.contract_id, i) not in definition_exist:
                                definition_ids.append(contract_feature.definition_ids[i])
                                definition_type_ids.append(contract_feature.definition_type_ids[i])
                                definition_attention_mask.append(contract_feature.definition_attention_mask[i])
                                definition_exist[(contract_feature.contract_id, i)] = total_lens
                                contract_definition_map_segment.append(total_lens)
                                total_lens += 1
                            else:
                                contract_definition_map_segment.append(definition_exist[(contract_feature.contract_id, i)])
                    contract_definition_map_one.append(contract_definition_map_segment)
                contract_definition_map.append(contract_definition_map_one)


            definition_ids = torch.tensor(definition_ids, dtype=torch.long)
            definition_type_ids = torch.tensor(definition_type_ids, dtype=torch.long)
            definition_attention_mask = torch.tensor(definition_attention_mask, dtype=torch.long)
            max_definition_map_num = max([len(c) for b in contract_definition_map for c in b])
            contract_definition_map = torch.tensor([[c + [0] * (max_definition_map_num - len(c)) for c in b] + [
                [0] * max_definition_map_num for _ in range(max_doc_num - len(b))] for b in contract_definition_map],
                                                   dtype=torch.long)
            #contract
            contract_input_ids = torch.tensor([[doc.input_ids for doc in contract_features] +
                                               [[self.tokenizer.pad_token_id] * max_doc_length
                                                for _ in range(max_doc_num - len(contract_features))] for contract_features in contracts_features], dtype=torch.long)
            contract_attention_mask = torch.tensor([[doc.attention_mask for doc in contract_features] +
                                               [[0] * max_doc_length
                                                for _ in range(max_doc_num - len(contract_features))] for contract_features in contracts_features], dtype=torch.long)
            contract_token_type_ids = torch.tensor([[doc.token_type_ids for doc in contract_features]+
                                               [[0] * max_doc_length
                                                for _ in range(max_doc_num - len(contract_features))] for contract_features in contracts_features], dtype=torch.long)
            contract_eval_ids = torch.tensor([[doc.eval_id for doc in contract_features] +
                                     [-1 for _ in range(max_doc_num - len(contract_features))] for
                                     contract_features in contracts_features], dtype=torch.long)
            contract_label_ids = torch.tensor([[doc.label_id for doc in contract_features] +
                                               [self.query_size for _ in range(max_doc_num - len(contract_features))]
                                               for
                                               contract_features in contracts_features], dtype=torch.long)
            if self.is_training:
                contract_start_position = torch.tensor([[doc.start_position for doc in contract_features] +
                                               [0 for _ in range(max_doc_num - len(contract_features))] for contract_features in contracts_features], dtype=torch.long)
                contract_end_position = torch.tensor([[doc.end_position for doc in contract_features] +
                                               [0 for _ in range(max_doc_num - len(contract_features))] for contract_features in contracts_features], dtype=torch.long)

                yield (definition_ids, definition_type_ids, definition_attention_mask, contract_definition_map,
                       contract_input_ids, contract_attention_mask, contract_token_type_ids, contract_start_position,
                       contract_end_position, contract_eval_ids, contract_label_ids)
            else:
                yield (definition_ids, definition_type_ids, definition_attention_mask, contract_definition_map,
                       contract_input_ids, contract_attention_mask, contract_token_type_ids, contract_eval_ids, contract_label_ids)
