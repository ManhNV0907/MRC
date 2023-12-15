from torch.utils.data import Dataset
from typing import List, Dict
import numpy as np
import torch
from torch import Tensor


class NERMRCDataset(Dataset):
    def __init__(self, data: List[Dict]):
        super(NERMRCDataset, self).__init__()
        self.data: List[Dict] = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        instance: Dict = self.data[item]
        input_ids: List[int] = instance["input_ids"]
        token_type_ids: List[int] = instance["token_type_ids"]
        attention_mask: List[int] = instance["attention_mask"]
        start_label_mask = instance["start_label_mask"]
        end_label_mask = instance["end_label_mask"]
        label = instance["label"]
        start_label: np.ndarray = np.zeros(shape=(len(input_ids),))
        end_label: np.ndarray = np.zeros(shape=(len(input_ids),))
        match_label: np.ndarray = np.zeros(shape=(len(input_ids), len(input_ids)))
        for start_index, end_index in instance["answer_token_indexes"]:
            start_label[start_index] = 1
            end_label[end_index] = 1
            match_label[start_index][end_index] = 1
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            "start_label": torch.tensor(start_label, dtype=torch.float),
            "end_label": torch.tensor(end_label, dtype=torch.float),
            "match_label": torch.tensor(match_label, dtype=torch.float),
            "start_label_mask": torch.ByteTensor(start_label_mask),
            "end_label_mask": torch.ByteTensor(end_label_mask),
            "label": label
        }


def collate_func(batch_data: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    max_length: int = max(map(lambda x: x["input_ids"].shape[0], batch_data))
    list_input_ids: List[Tensor] = []
    list_token_type_ids: List[Tensor] = []
    list_attention_mask: List[Tensor] = []
    list_start_label: List[Tensor] = []
    list_end_label: List[Tensor] = []
    list_match_label: List[Tensor] = []
    list_start_label_mask: List[Tensor] = []
    list_end_label_mask: List[Tensor] = []
    list_label = []
    for instance in batch_data:
        length: int = instance["input_ids"].shape[0]

        input_ids = torch.zeros(size=(max_length,), dtype=torch.long)
        input_ids[:length] = instance["input_ids"]
        list_input_ids.append(input_ids)

        token_type_ids = torch.zeros(size=(max_length,), dtype=torch.long)
        token_type_ids[:length] = instance["token_type_ids"]
        list_token_type_ids.append(token_type_ids)

        attention_mask = torch.zeros(size=(max_length,), dtype=torch.float)
        attention_mask[:length] = instance["attention_mask"]
        list_attention_mask.append(attention_mask)

        start_label = torch.zeros(size=(max_length,), dtype=torch.float)
        start_label[:length] = instance["start_label"]
        list_start_label.append(start_label)

        end_label = torch.zeros(size=(max_length,), dtype=torch.float)
        end_label[:length] = instance["end_label"]
        list_end_label.append(end_label)

        match_label = torch.zeros(size=(max_length, max_length), dtype=torch.float)
        match_label[:length, :length] = instance["match_label"]
        list_match_label.append(match_label)

        start_label_mask = torch.zeros(size=(max_length,), dtype=torch.uint8)
        start_label_mask[:length] = instance["start_label_mask"]
        list_start_label_mask.append(start_label_mask)

        end_label_mask = torch.zeros(size=(max_length,), dtype=torch.uint8)
        end_label_mask[:length] = instance["end_label_mask"]
        list_end_label_mask.append(end_label_mask)

        list_label.append(instance["label"])
        # print(list_label)

    return {
        "input_ids": torch.stack(list_input_ids, dim=0),
        "token_type_ids": torch.stack(list_token_type_ids, dim=0),
        "attention_mask": torch.stack(list_attention_mask, dim=0),
        "start_label": torch.stack(list_start_label, dim=0),
        "end_label": torch.stack(list_end_label, dim=0),
        "match_label": torch.stack(list_match_label, dim=0),
        "start_label_mask": torch.stack(list_start_label_mask, dim = 0),
        "end_label_mask": torch.stack(list_end_label_mask, dim=0),
        "label": list_label

    }
