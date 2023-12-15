import torch.nn as nn
from torch import Tensor
from typing import Tuple
import torch


class NERMRCLoss(nn.Module):
    def __init__(self):
        super(NERMRCLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def get_start_or_end_loss(self, logit: Tensor, target: Tensor, token_type_ids: Tensor):
        """
        :param logit: batch_size, length
        :param target: batch_size, length
        :param token_type_ids: batch_size, length
        :return:
        """
        batch_size, length = logit.shape

        is_predict_true = logit > 0    # batch_size, length
        is_target_true = target > 0    # batch_size, length

        valid_mask = token_type_ids.bool()

        mask = (is_predict_true | is_target_true) & valid_mask    # batch_size, length

        mask = mask.reshape(batch_size*length).float()    # batch_size * length
        logit = logit.reshape(batch_size*length)    # batch_size * length
        target = target.reshape(batch_size*length)   # batch_size * length

        loss = self.bce_loss(logit, target)    # batch_size * length
        loss = loss * mask    # batch_size * length
        loss = loss.sum() / (mask.sum() + 1e-8)

        return loss

    def get_match_loss(self, logit: Tensor, target: Tensor, token_type_ids: Tensor):
        """

        :param logit: batch_size, length, length
        :param target: batch_size, length, length
        :param token_type_ids: batch_size, length
        :return:
        """
        batch_size, length = logit.shape[:2]

        is_predict_true = logit > 0    # batch_size, length, length
        is_target_true = target > 0    # batch_size, length, length

        start_token_type_ids = token_type_ids.unsqueeze(dim=2).repeat(1, 1, length).bool()    # batch_size, length, length
        end_token_type_ids = token_type_ids.unsqueeze(dim=1).repeat(1, length, 1).bool()     # batch_size, length, length
        order_mask = torch.triu(torch.ones(size=(length, length))).unsqueeze(dim=0).repeat(batch_size, 1, 1).bool()    # batch_size, length, length
        valid_mask = start_token_type_ids & end_token_type_ids & order_mask.to(logit.device)    # batch_size, length, length

        mask = (is_predict_true | is_target_true) & valid_mask    # batch_size, length, length

        mask = mask.reshape(batch_size * length * length).float()  # batch_size * length * length
        logit = logit.reshape(batch_size * length * length)  # batch_size * length * length
        target = target.reshape(batch_size * length * length)  # batch_size * length * length

        loss = self.bce_loss(logit, target)  # batch_size * length * length
        loss = loss * mask  # batch_size * length * length
        loss = loss.sum() / (mask.sum() + 1e-8)

        return loss

    def forward(self, start_logit: Tensor, end_logit: Tensor, match_logit: Tensor,
                start_target: Tensor, end_target: Tensor, match_target: Tensor,
                token_type_ids: Tensor):
        start_loss = self.get_start_or_end_loss(logit=start_logit, target=start_target, token_type_ids=token_type_ids)
        end_loss = self.get_start_or_end_loss(logit=end_logit, target=end_target, token_type_ids=token_type_ids)
        match_loss = self.get_match_loss(logit=match_logit, target=match_target, token_type_ids=token_type_ids)

        return start_loss + end_loss + match_loss
