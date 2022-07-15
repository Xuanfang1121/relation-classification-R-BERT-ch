# -*- coding: utf-8 -*-
# @Time    : 2022/7/11 23:47
# @Author  : zxf
import json
import random

import torch
import numpy as np
from transformers import AutoConfig
from transformers import BertConfig
from transformers import AutoTokenizer
from transformers import BertTokenizer

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels


ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def load_tokenizer(pretrain_model_path, pretrain_model_type):
    """加载tokenzier添加特殊符号"""
    if "bert" in pretrain_model_type:
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def get_pretrain_config(pretrain_model_path, pretrain_model_type,
                        label2id, id2label):
    if "bert" in pretrain_model_type:
        config = BertConfig.from_pretrained(pretrain_model_path,
                                            num_labels=len(label2id),
                                            id2label=id2label,
                                            label2id=label2id
                                            )
    else:
        config = AutoConfig.from_pretrained(pretrain_model_path,
                                            num_labels=len(label2id),
                                            id2label=id2label,
                                            label2id=label2id)
    return config


def get_label(label_file):
    with open(label_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def set_global_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch_data):
    input_ids = [feature.input_ids for feature in batch_data]
    attention_mask = [feature.attention_mask for feature in batch_data]
    token_type_ids = [feature.token_type_ids for feature in batch_data]
    label_id = [feature.label_id for feature in batch_data]
    e1_mask = [feature.e1_mask for feature in batch_data]
    e2_mask = [feature.e2_mask for feature in batch_data]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    label_id = torch.tensor(label_id, dtype=torch.long)
    e1_mask = torch.tensor(e1_mask, dtype=torch.long)
    e2_mask = torch.tensor(e2_mask, dtype=torch.long)
    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": label_id,
            "e1_mask": e1_mask,
            "e2_mask": e2_mask}


def model_evaluate(model, dev_dataloader, device, id2label, logger):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for step, batch_data in enumerate(dev_dataloader):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            token_type_ids = batch_data["token_type_ids"].to(device)
            labels = batch_data["labels"].to(device)
            e1_mask = batch_data["e1_mask"].to(device)
            e2_mask = batch_data["e2_mask"].to(device)
            loss, logits = model(input_ids, attention_mask, token_type_ids,
                                 e1_mask, e2_mask, labels)
            pred = torch.softmax(logits, dim=1)
            pred = torch.argmax(pred, dim=1).data.cpu().numpy().tolist()
            labels = labels.data.cpu().numpy().tolist()
            pred_labels.extend(pred)
            true_labels.extend(labels)

    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0, average='macro')
    recall = recall_score(true_labels, pred_labels, zero_division=0, average='macro')
    f1 = f1_score(true_labels, pred_labels, zero_division=0, average='macro')
    pred_label_list = unique_labels(true_labels, pred_labels).tolist()
    target_names = [id2label[str(item)] for item in pred_label_list]
    logger.info(classification_report(true_labels, pred_labels, target_names=target_names))
    return acc, recall, precision, f1


def save_file(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))