import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple
from transformers import BertModel

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate=0.1, act_func="gelu", intermediate_hidden_size=1024):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size, self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == "gelu":
            features_output1 = F.gelu(features_output1)
        elif self.act_func == "relu":
            features_output1 = F.relu(features_output1)
        elif self.act_func == "tanh":
            features_output1 = F.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2

class NERMRCModel(nn.Module):
    def __init__(self, context_presenter: BertModel, hidden_dim: int):
        super(NERMRCModel, self).__init__()
        self.context_presenter: BertModel = context_presenter
        self.start_classifier = nn.Linear(in_features=hidden_dim, out_features=1)
        self.end_classifier = nn.Linear(in_features=hidden_dim, out_features=1)
        # self.match_classifier = nn.Linear(in_features=2*hidden_dim, out_features=1)
        self.match_classifier = MultiNonLinearClassifier(hidden_size=2*hidden_dim, num_label=1)


    def forward(self, input_ids: Tensor, token_type_ids: Tensor,
                attention_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """

        :param input_ids: batch_size, query_length + context_length
        :param token_type_ids: batch_size, query_length + context_length
        :param attention_mask: batch_size, query_length + context_length
        :return:
        """
        length = input_ids.shape[1]
        context = self.context_presenter(input_ids=input_ids, token_type_ids=token_type_ids,
                                         attention_mask=attention_mask).last_hidden_state     # batch_size, query_length + context_length, hidden_features

        start_logit = self.start_classifier(context).squeeze(dim=2)    # batch_size, query_length + context_length
        end_logit = self.end_classifier(context).squeeze(dim=2)     # batch_size, query_length + context_length

        start_context_expand = context.unsqueeze(dim=2).repeat(1, 1, length, 1)    # batch_size, query_length + context_length, query_length + context_length, hidden_features
        end_context_expand = context.unsqueeze(dim=1).repeat(1, length, 1, 1)    # batch_size, query_length + context_length, query_length + context_length, hidden_features
        context_concat = torch.cat([start_context_expand, end_context_expand], dim=3)    # batch_size, query_length + context_length, query_length + context_length, 2*hidden_features
        match_logit = self.match_classifier(context_concat).squeeze(dim=3)    # batch_size, query_length + context_length, query_length + context_length

        return start_logit, end_logit, match_logit
