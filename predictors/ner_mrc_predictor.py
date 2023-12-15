from models import NERMRCModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import torch
from config import MAX_LENGTH
from torch import Tensor
from typing import List, Dict, Tuple


class NERMRCPredictor:
    def __init__(self, model: NERMRCModel, tokenizer: BertTokenizerFast,
                 device: torch.device, label_to_query: Dict[str, str]):
        self.model: NERMRCModel = model
        self.tokenizer: BertTokenizerFast = tokenizer
        self.device: torch.device = device
        self.label_to_query: Dict[str, str] = label_to_query

    def get_tokenize_result_for_query_context(self, label: str, query: str, context: str) -> Dict:
        tokenize_result: Dict = dict(self.tokenizer(
            text=query, text_pair=context, max_length=MAX_LENGTH,
            add_special_tokens=True, truncation=True, return_token_type_ids=True,
            return_offsets_mapping=True
        ))
        tokenize_result["query"] = query
        tokenize_result["context"] = context
        tokenize_result["label"] = label
        return tokenize_result

    def auto_padding(self, list_tensors: List[Tensor], tensor_type: torch.dtype) -> Tensor:
        max_length: int = max(map(lambda x: x.shape[0], list_tensors))
        result: List[Tensor] = []
        for tensor in list_tensors:
            length: int = tensor.shape[0]
            pad_tensor = torch.zeros(size=(max_length,), dtype=tensor_type)
            pad_tensor[:length] = tensor
            result.append(pad_tensor)
        return torch.stack(result, dim=0)

    def get_character_offsets(self, tokenize_result: Dict, start_logit: Tensor, end_logit: Tensor,
                              match_logit: Tensor) -> List[Tuple[int, int]]:
        length: int = len(tokenize_result["input_ids"])
        start_logit = start_logit[:length]
        end_logit = end_logit[:length]
        match_logit = match_logit[:length, :length]

        is_start = start_logit > 0    # length
        is_start = is_start.unsqueeze(dim=1).repeat(1, length)     # length, length
        is_end = end_logit > 0    # length
        is_end = is_end.unsqueeze(dim=0).repeat(length, 1)    # length, length
        is_match = match_logit > 0    # length, length

        order_mask = torch.triu(torch.ones(size=(length, length))).bool().to(start_logit.device)    # length, length

        token_type_ids = torch.tensor(tokenize_result["token_type_ids"]).bool().to(start_logit.device)   # length
        start_token_type_ids = token_type_ids.unsqueeze(dim=1).repeat(1, length)   # length, length
        end_token_type_ids = token_type_ids.unsqueeze(dim=0).repeat(length, 1)    # length, length

        is_predict = is_start & is_end & is_match & order_mask & start_token_type_ids & end_token_type_ids

        result: List[Tuple[int, int]] = []
        offset_mapping: List[Tuple[int, int]] = tokenize_result["offset_mapping"]
        for i in range(length):
            for j in range(i, length):
                if is_predict[i][j].item():
                    result.append((offset_mapping[i][0], offset_mapping[j][1]))
        return result

    def predict_one(self, text: str) -> List[Dict]:
        tokenize_results: List[Dict] = [
            self.get_tokenize_result_for_query_context(label=label, query=query, context=text)
            for label, query in self.label_to_query.items()
        ]

        input_ids: List[Tensor] = list(map(lambda x: torch.tensor(x['input_ids'], dtype=torch.long),
                                           tokenize_results))
        input_ids: Tensor = self.auto_padding(list_tensors=input_ids, tensor_type=torch.long)
        token_type_ids: List[Tensor] = list(map(lambda x: torch.tensor(x['token_type_ids'], dtype=torch.long),
                                                tokenize_results))
        token_type_ids: Tensor = self.auto_padding(list_tensors=token_type_ids, tensor_type=torch.long)
        attention_mask: List[Tensor] = list(map(lambda x: torch.tensor(x['attention_mask'], dtype=torch.float),
                                                tokenize_results))
        attention_mask: Tensor = self.auto_padding(list_tensors=attention_mask, tensor_type=torch.float)

        predict_result: List[Dict] = []
        with torch.no_grad():
            self.model.eval()
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            start_logit, end_logit, match_logit = self.model(input_ids=input_ids,
                                                             token_type_ids=token_type_ids,
                                                             attention_mask=attention_mask)
            for index, tokenize_result in enumerate(tokenize_results):
                offsets: List[Tuple[int, int]] = self.get_character_offsets(
                    tokenize_result=tokenize_result, start_logit=start_logit[index],
                    end_logit=end_logit[index], match_logit=match_logit[index]
                )
                if len(offsets) > 0:
                    predict_result.append({
                        "label": tokenize_result["label"],
                        "offsets": offsets
                    })
            del input_ids, token_type_ids, attention_mask, start_logit, end_logit, match_logit
            torch.cuda.empty_cache()
        return predict_result
