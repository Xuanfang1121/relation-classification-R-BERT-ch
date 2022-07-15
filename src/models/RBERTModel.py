# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    def __init__(self, config, dropout_rate, hidden_type, pooler_type):
        super(RBERT, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.num_labels = config.num_labels
        self.dropout_rate = dropout_rate

        self.cls_fc_layer = FCLayer(config.hidden_size,
                                    config.hidden_size,
                                    self.dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size,
                                       config.hidden_size,
                                       self.dropout_rate)
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            config.num_labels,
            self.dropout_rate,
            use_activation=False,
        )
        self.hidden_type = hidden_type # "last2_hidden_state"
        # self.type = "first_last_hidden_state"
        # self.type = "last_hidden_state"
        self.pooler_type = pooler_type

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, labels=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            output_hidden_states=True,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output = outputs[0]
        if self.pooler_type == "pooler":
            pooled_output = outputs[1]  # [CLS]
        elif self.pooler_type == "cls":
            pooled_output = outputs["last_hidden_state"][:, 0]
        if self.hidden_type == "last_hidden_state":
            sequence_output = outputs["last_hidden_state"]
        elif self.hidden_type == "last2_hidden_state":
            # output = outputs["hidden_states"][:-2].mean(dim=1)
            sequence_output = (outputs["hidden_states"][-1] + outputs["hidden_states"][-2])
        elif self.hidden_type == "first_last_hidden_state":
            sequence_output = (outputs["hidden_states"][1] + outputs["hidden_states"][-1])

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        # print("logits shape: ", logits.shape)
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
                # loss = loss_fct(logits.flatten(1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                # print("label: ", labels.shape)
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits