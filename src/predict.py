# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 22:53
# @Author  : zxf
import os
import json

import torch
from transformers import BertConfig

from models.RBERTModel import RBERT
from utils.util import load_tokenizer


def get_infer_sent(sent, entity1, entity2):
    if entity1 in sent and entity2 in sent:
        entity1_index = sent.index(entity1)
        entity2_index = sent.index(entity2)
        sent = list(sent)
        if entity2_index >= entity1_index:
            sent.insert(entity1_index, '<e1>')
            sent.insert(entity1_index + len(entity1) + 1, '</e1>')
            sent.insert(entity2_index + 2, '<e2>')
            sent.insert(entity2_index + len(entity2) + 3, '</e2>')
        else:
            sent.insert(entity2_index, '<e2>')
            sent.insert(entity2_index + len(entity2) + 1, '</e2>')
            sent.insert(entity1_index + 2, '<e1>')
            sent.insert(entity1_index + len(entity1) + 3, '</e1>')
        sent = ''.join(sent)
    else:
        sent = None
        print("给定的句子中不包含实体")
    return sent


def get_inference_data_feature(sent, tokenizer, max_seq_len, device):
    tokens_a = tokenizer.tokenize(sent)

    if len(tokens_a) >= max_seq_len - 1:
        tokens_a = tokens_a[: (max_seq_len - 2)]
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
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * len(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(
            len(input_ids), max_seq_len)

        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(device)
        e1_mask = torch.tensor([e1_mask], dtype=torch.long).to(device)
        e2_mask = torch.tensor([e2_mask], dtype=torch.long).to(device)
        return input_ids, attention_mask, token_type_ids, e1_mask, e2_mask


def predict(sent, entity1, entity2, model_path,
            pretrain_model_path, pretrain_model_type, max_seq_len,
            dropout_rate, hidden_type, pooler_type, label_file, device):
    # get model input sent
    sent = get_infer_sent(sent, entity1, entity2)
    if sent is not None:
        # label
        with open(label_file, "r", encoding="utf-8") as f:
            label = [line.strip() for line in f.readlines()]
        label2id = {key: i for i, key in enumerate(label)}
        id2label = {value: key for key, value in label2id.items()}
        tokenizer = load_tokenizer(pretrain_model_path, pretrain_model_type)

        # get model
        # config = torch.load(os.path.join(model_config_file))
        config = BertConfig.from_pretrained(model_path)
        model = RBERT.from_pretrained(model_path,
                                      config=config,
                                      dropout_rate=dropout_rate,
                                      hidden_type=hidden_type,
                                      pooler_type=pooler_type)
        model.to(device)
        model.eval()

        # get sent feature
        input_ids, attention_mask, token_type_ids, e1_mask, \
        e2_mask = get_inference_data_feature(sent, tokenizer, max_seq_len, device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask)
            pred = torch.softmax(logits, dim=1)
            pred = torch.argmax(pred, dim=1).data.cpu().numpy().tolist()
            predlabel = [id2label[item] for item in pred]

        result = {"text": sent,
                  "entity1": entity1,
                  "entity2": entity2,
                  "relation": predlabel}

    else:
        result = {"text": sent,
                  "entity1": entity1,
                  "entity2": entity2,
                  "relation": []}

    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sent = "在宁冈茅坪，毛泽覃见到日夜思念的大哥毛泽东，向他介绍了南昌起义军余部的详细情况及朱德、陈毅派他来井冈山"
    entity1 = "毛泽东"
    entity2 = "朱德"
    model_path = "./output/"
    pretrain_model_path = "D:/Spyder/pretrain_model/transformers_torch_tf/chinese-roberta-wwm-ext/"
    pretrain_model_type = "bert-base"
    max_seq_len = 128
    dropout_rate = 0.1
    hidden_type = "last_hidden_state"
    pooler_type = "pooler"
    label_file = "./data/relation_data/relation.txt"
    device = "cpu"
    result = predict(sent, entity1, entity2, model_path,
            pretrain_model_path, pretrain_model_type, max_seq_len,
            dropout_rate, hidden_type, pooler_type, label_file, device)
    print(result)