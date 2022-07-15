# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 22:39
# @Author  : zxf
import csv
import copy
import json

from common.common import logger


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id,
                 e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Processor for the Semeval data set """

    def __init__(self, data_file, labels, max_seq_len, tokenizer, mode):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.relation_labels = labels
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.examples = self.get_examples(data_file, mode)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = self.relation_labels.index(line[0])
            tokens = self.tokenizer.tokenize(text_a)[:self.max_seq_len - 2]
            if "</e1>" in tokens and "</e2>" in tokens:
                examples.append(InputExample(guid=guid, text_a=text_a, label=label))
            if i % 1000 == 0:
                logger.info("read data processing:{}/{}".format(i, len(lines)))
        return examples

    def get_examples(self, data_file, mode):
        """
        Args:
            mode: train, dev, test
        """

        data = self._read_tsv(data_file)
        examples = self._create_examples(data, mode)
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        tokens_a = self.tokenizer.tokenize(example.text_a)

        if len(tokens_a) >= self.max_seq_len - 1:
            tokens_a = tokens_a[: (self.max_seq_len - 2)]
        # 句子按照最大的长度截取完，需要确认关系抽取的两个实体仍然在句子中, 我新增的判断20210327
        if "</e1>" in tokens_a and "</e2>" in tokens_a:
            e11_p = tokens_a.index("<e1>")  # the start position of entity1
            e12_p = tokens_a.index("</e1>")  # the end position of entity1
            e21_p = tokens_a.index("<e2>")  # the start position of entity2
            e22_p = tokens_a.index("</e2>")  # the end position of entity2

            # Replace the token
            tokens_a[e11_p] = "$"
            tokens_a[e12_p] = "$"
            tokens_a[e21_p] = "#"
            tokens_a[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            # 获取数据特征
            tokens = tokens_a
            tokens = [self.cls_token] + tokens + [self.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_type_ids = [0] * len(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.max_seq_len - len(input_ids)
            input_ids = input_ids + ([self.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)

            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            assert len(input_ids) == self.max_seq_len, "Error with input length {} vs {}".format(
                len(input_ids), self.max_seq_len)

            assert len(attention_mask) == self.max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), self.max_seq_len
            )
            assert len(token_type_ids) == self.max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), self.max_seq_len
            )

            label_id = int(example.label)

            features = InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label_id=label_id,
                    e1_mask=e1_mask,
                    e2_mask=e2_mask,
                )

        return features

    def convert_examples_to_features(self, examples):
        """数据特征处理，转化为模型输入特征"""
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            # 获得tokens 注意这里没有cls 和 sep
            tokens_a = self.tokenizer.tokenize(example.text_a)

            if len(tokens_a) >= self.max_seq_len - 1:
                tokens_a = tokens_a[: (self.max_seq_len - 2)]
            # 句子按照最大的长度截取完，需要确认关系抽取的两个实体仍然在句子中, 我新增的判断20210327
            if "</e1>" in tokens_a and "</e2>" in tokens_a:
                e11_p = tokens_a.index("<e1>")  # the start position of entity1
                e12_p = tokens_a.index("</e1>")  # the end position of entity1
                e21_p = tokens_a.index("<e2>")  # the start position of entity2
                e22_p = tokens_a.index("</e2>")  # the end position of entity2

                # Replace the token
                tokens_a[e11_p] = "$"
                tokens_a[e12_p] = "$"
                tokens_a[e21_p] = "#"
                tokens_a[e22_p] = "#"

                # Add 1 because of the [CLS] token
                e11_p += 1
                e12_p += 1
                e21_p += 1
                e22_p += 1

                # 获取数据特征
                tokens = tokens_a
                tokens = [self.cls_token] + tokens + [self.sep_token]
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                token_type_ids = [0] * len(tokens)
                # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
                attention_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding_length = self.max_seq_len - len(input_ids)
                input_ids = input_ids + ([self.pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)

                # e1 mask, e2 mask
                e1_mask = [0] * len(attention_mask)
                e2_mask = [0] * len(attention_mask)

                for i in range(e11_p, e12_p + 1):
                    e1_mask[i] = 1
                for i in range(e21_p, e22_p + 1):
                    e2_mask[i] = 1

                assert len(input_ids) == self.max_seq_len, "Error with input length {} vs {}".format(
                    len(input_ids), self.max_seq_len)

                assert len(attention_mask) == self.max_seq_len, "Error with attention mask length {} vs {}".format(
                    len(attention_mask), self.max_seq_len
                )
                assert len(token_type_ids) == self.max_seq_len, "Error with token type length {} vs {}".format(
                    len(token_type_ids), self.max_seq_len
                )

                label_id = int(example.label)

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % example.guid)
                    logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                    logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                    logger.info("label: %s (id = %d)" % (example.label, label_id))
                    logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
                    logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

                features.append(
                    InputFeatures(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        label_id=label_id,
                        e1_mask=e1_mask,
                        e2_mask=e2_mask,
                    )
                )

        return features