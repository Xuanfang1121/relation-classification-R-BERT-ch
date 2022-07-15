# -*- coding: utf-8 -*-
# @Time    : 2022/7/12 23:18
# @Author  : zxf
import os
import traceback

import torch
# from torch.optim import AdamW
from transformers import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from common.common import logger
from utils.util import get_label
from utils.util import collate_fn
from models.RBERTModel import RBERT
from utils.util import load_tokenizer
from utils.util import model_evaluate
from config.getConfig import get_config
from utils.util import get_pretrain_config
from utils.util import set_global_random_seed
from utils.dataprocessor import DataProcessor


def train(config_init):
    Config = get_config(config_init)
    os.environ["CUDA_VISIBLE_DEVICES"] = Config["visible_gpus"]
    device = "cpu" if Config["visible_gpus"] == "-1" else "cuda"
    # check path
    if not os.path.exists(Config["output_path"]):
        os.mkdir(Config["output_path"])

    # seed
    set_global_random_seed(Config["seed"])
    # load tokenizer
    tokenizer = load_tokenizer(Config["pretrain_model_path"],
                               Config["pretrain_model_type"])
    tokenizer.save_pretrained(Config["output_path"])
    # get label
    rel_labels = get_label(Config["rel_file"])
    rel2id = {value: i for i, value in enumerate(rel_labels)}
    id2rel = {str(value): key for key, value in rel2id.items()}

    # get dataset
    train_dataset = DataProcessor(Config["train_data_path"],
                                  rel_labels, Config["max_seq_length"], tokenizer, "train")
    dev_dataset = DataProcessor(Config["dev_data_path"],
                                rel_labels, Config["max_seq_length"], tokenizer, "train")
    logger.info("train dataset:{}".format(len(train_dataset)))
    logger.info("dev dataset:{}".format(len(dev_dataset)))

    # get dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=Config["batch_size"],
                                  shuffle=False, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=Config["dev_batch_size"],
                                shuffle=False, collate_fn=collate_fn)
    logger.info("pre epoch step:{}".format(len(train_dataloader)))

    # get config
    pretrain_config = get_pretrain_config(Config["pretrain_model_path"],
                                          Config["pretrain_model_type"],
                                          rel2id, id2rel)
    model = RBERT.from_pretrained(Config["pretrain_model_path"],
                                  config=pretrain_config,
                                  dropout_rate=Config["dropout_rate"],
                                  hidden_type=Config["hidden_type"],
                                  pooler_type=Config["pooler_type"])
    model.to(device)

    if Config["max_steps"] > 0:
        t_total = Config["max_steps"]
        epochs = (
                Config["max_steps"] // (len(train_dataloader) // Config["gradient_accumulation_steps"]) + 1
        )
    else:
        t_total = len(train_dataloader) // Config["gradient_accumulation_steps"] * Config["epochs"]
    logger.info("total optimizer step:{}".format(t_total))
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": Config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=Config["learning_rate"],
        eps=Config["adam_epsilon"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config["warmup_steps"],
        num_training_steps=t_total,
    )

    best_f1 = 0
    # todo 验证一下optimizer loss这些顺序写的对不
    model.zero_grad()

    for epoch in range(Config["epochs"]):
        model.train()
        for step, batch_data in enumerate(train_dataloader):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            token_type_ids = batch_data["token_type_ids"].to(device)
            labels = batch_data["labels"].to(device)
            e1_mask = batch_data["e1_mask"].to(device)
            e2_mask = batch_data["e2_mask"].to(device)
            loss, _ = model(input_ids, attention_mask, token_type_ids,
                                 e1_mask, e2_mask, labels)
            #

            if Config["gradient_accumulation_steps"] > 1:
                loss = loss / Config["gradient_accumulation_steps"]
            optimizer.zero_grad()
            loss.backward()

            if (step + 1) % Config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               Config["max_grad_norm"])

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            # model.zero_grad()

            if (step + 1) % Config["pre_epoch_step_print"] == 0:
                logger.info("epoch:{}/{}, step:{}/{}, loss:{}".format(epoch + 1,
                                                                      Config["epochs"],
                                                                      step + 1,
                                                                      len(train_dataloader),
                                                                      loss))
        logger.info("model evaluate")
        acc, recall, precision, f1 = model_evaluate(model, dev_dataloader, device, id2rel, logger)
        logger.info("acc:{}, recall:{}, precision:{}, f1:{}".format(acc, recall, precision, f1))

        if f1 >= best_f1:
            best_f1 = f1
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(Config["output_path"])

            # Save training arguments together with the trained model
            torch.save(Config, os.path.join(Config["output_path"], "training_args.bin"))


if __name__ == "__main__":
    config_init = "./config/config.ini"
    train(config_init)
