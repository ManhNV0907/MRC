from typing import Dict, List, Tuple, Set
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from config import BERT_CASED_TOKENIZER_DIR, MAX_LENGTH
from tqdm import tqdm
from common.utils import JsonReadObjectFromLocalPatient, JsonWriteObjectToLocalPatient
from common.logger import SingletonLogger


SingletonLogger.set_logger_name(name="Process dataset")


TOKENIZER: BertTokenizerFast = AutoTokenizer.from_pretrained(
    "bert-base-uncased", cache_dir=BERT_CASED_TOKENIZER_DIR, use_fast=True
)
JSON_READER = JsonReadObjectFromLocalPatient()
JSON_WRITER = JsonWriteObjectToLocalPatient()


def read_data_from_file(file_name: str) -> List[Dict]:
    result: List[Dict] = []
    with open(file_name, mode="r", buffering=10000) as file_obj:
        current_words: List[str] = []
        current_labels: List[str] = []
        for line in file_obj:
            line: str = line.strip()
            if line == "-DOCSTART- -X- -X- O":
                continue
            if len(line) == 0:
                if len(current_words) > 0:
                    result.append({
                        "words": current_words,
                        "labels": current_labels
                    })
                current_words = []
                current_labels = []
            else:
                word, _, _, label = line.split(" ")
                current_words.append(word)
                current_labels.append(label)
        if len(current_words) > 0:
            result.append({
                "words": current_words,
                "labels": current_labels
            })
    return result


def convert_data_to_string_and_offset(data: List[Dict]) -> List[Dict]:
    result: List[Dict] = []
    for instance in data:
        text: str = ""
        label_to_offsets: Dict[str, List[Tuple[int, int]]] = {}
        current_label: str = ""
        current_start_offset: int = -1
        current_end_offset: int = -1
        for word, label in zip(instance["words"], instance["labels"]):
            if label == "O":
                if len(current_label) > 0:
                    if current_label not in label_to_offsets:
                        label_to_offsets[current_label] = []
                    label_to_offsets[current_label].append((current_start_offset, current_end_offset))
                current_label = ""
                current_start_offset = -1
                current_end_offset = -1
            elif label.startswith("B"):
                if len(current_label) > 0:
                    if current_label not in label_to_offsets:
                        label_to_offsets[current_label] = []
                    label_to_offsets[current_label].append((current_start_offset, current_end_offset))
                current_label = label[2:]
                add_space: int = 0 if len(text) == 0 else 1
                current_start_offset = len(text) + add_space
                current_end_offset = len(text) + add_space + len(word)
            elif label.startswith("I"):
                current_end_offset = len(text) + 1 + len(word)
            text = word if len(text) == 0 else text + " " + word
        if len(current_label) > 0:
            if current_label not in label_to_offsets:
                label_to_offsets[current_label] = []
            label_to_offsets[current_label].append((current_start_offset, current_end_offset))
        result.append({
            "text": text,
            "label_to_offsets": label_to_offsets
        })
    return result


def get_set_offset(offsets: List[Tuple[int, int]]) -> Tuple[Set[int], Set[int]]:
    start_offsets_set: Set[int] = set()
    end_offsets_set: Set[int] = set()
    for start_offset, end_offset in offsets:
        start_offsets_set.add(start_offset)
        end_offsets_set.add(end_offset)
    return start_offsets_set, end_offsets_set


def get_offset_to_token_index(
        token_type_ids: List[int], offset_mapping: List[Tuple[int, int]],
        start_offsets_set: Set[int], end_offsets_set: Set[int]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    start_offset_to_token_index: Dict[int, int] = {}
    end_offset_to_token_index: Dict[int, int] = {}
    for index, token_type_id in enumerate(token_type_ids):
        if token_type_id == 0:
            continue
        start_offset, end_offset = offset_mapping[index]
        if start_offset == 0 and end_offset == 0:
            continue
        if start_offset in start_offsets_set:
            start_offset_to_token_index[start_offset] = index
        if end_offset in end_offsets_set:
            end_offset_to_token_index[end_offset] = index
    return start_offset_to_token_index, end_offset_to_token_index


def convert_offset_to_token_id(
        offsets: List[Tuple[int, int]],
        start_offset_to_token_index: Dict[int, int],
        end_offset_to_token_index: Dict[int, int]
) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    for start_offset, end_offset in offsets:
        start_token_index: int = start_offset_to_token_index.get(start_offset, None)
        end_token_index: int = end_offset_to_token_index.get(end_offset, None)
        if start_token_index is not None and end_token_index is not None:
            result.append((start_token_index, end_token_index))
    return result


def get_entity_token_indexes(
        offsets: List[Tuple[int, int]],
        token_type_ids: List[int], offset_mapping: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    start_offsets_set, end_offsets_set = get_set_offset(offsets=offsets)
    start_offset_to_token_index, end_offset_to_token_index = get_offset_to_token_index(
        token_type_ids=token_type_ids, offset_mapping=offset_mapping,
        start_offsets_set=start_offsets_set, end_offsets_set=end_offsets_set
    )
    token_indexes: List[Tuple[int, int]] = convert_offset_to_token_id(
        offsets=offsets, start_offset_to_token_index=start_offset_to_token_index,
        end_offset_to_token_index=end_offset_to_token_index
    )
    return token_indexes


def process_data(data: List[Dict], label_to_query: Dict[str, str]) -> List[Dict]:
    result: List[Dict] = []
    progress_bar = tqdm(data, desc="Processing data...")
    for instance in progress_bar:
        context: str = instance["text"]
        label_to_offsets: Dict[str, List[Tuple[int, int]]] = instance["label_to_offsets"]
        for label, query in label_to_query.items():
            tokenize_result: Dict = dict(TOKENIZER(
                text=query, text_pair=context, max_length=MAX_LENGTH,
                add_special_tokens=True, truncation=True, return_token_type_ids=True,
                return_offsets_mapping=True
            ))
            r = TOKENIZER(
                text=query, text_pair=context, max_length=MAX_LENGTH,
                add_special_tokens=True, truncation=True, return_token_type_ids=True,
                return_offsets_mapping=True
            ).word_ids()
            # print(r)
            # print(tokenize_result)

            seq_len = len(tokenize_result["input_ids"])
            offsets: List[Tuple[int, int]] = label_to_offsets.get(label, [])
            token_indexes: List[Tuple[int, int]] = get_entity_token_indexes(
                offsets=offsets, token_type_ids=tokenize_result["token_type_ids"],
                offset_mapping=tokenize_result["offset_mapping"]
            )
            label_mask = []
            for token_idx in range(seq_len):
                if (tokenize_result["token_type_ids"][token_idx] == 0) or \
                    (tokenize_result["offset_mapping"][token_idx] == (0, 0)):
                    label_mask.append(0)
                else:
                    label_mask.append(1)
            
            start_label_mask = label_mask.copy()
            end_label_mask = label_mask.copy()
            # print(label_mask)
            
            for token_idx in range(seq_len):
                # word_ids = tokenize_result["words"]
                word_ids = r
                curr_word_id = word_ids[token_idx]
                next_word_id = word_ids[token_idx + 1] if token_idx + 1 < seq_len else None
                prev_word_id = word_ids[token_idx - 1] if token_idx - 1 >= 0 else None
                if (prev_word_id is not None) and (curr_word_id == prev_word_id):
                    start_label_mask[token_idx] = 0
                if (next_word_id is not None) and (curr_word_id == next_word_id):
                    end_label_mask[token_idx] = 0

            tokenize_result["query"] = query
            tokenize_result["context"] = context
            tokenize_result["label"] = label
            tokenize_result["start_label_mask"] = start_label_mask
            tokenize_result["end_label_mask"] = end_label_mask
            tokenize_result["answer_token_indexes"] = token_indexes
            result.append(tokenize_result)
    progress_bar.close()
    return result


LABEL_TO_QUERY: Dict[str, str] = JSON_READER.read(file_name="data/raw_data/label_to_query.json")

DATA: List[Dict] = read_data_from_file(file_name="data/raw_data/train.txt")
DATA: List[Dict] = convert_data_to_string_and_offset(data=DATA)
DATA: List[Dict] = process_data(data=DATA, label_to_query=LABEL_TO_QUERY)
JSON_WRITER.write(x=DATA, file_name="data/clean_data/train.json")

DATA: List[Dict] = read_data_from_file(file_name="data/raw_data/valid.txt")
DATA: List[Dict] = convert_data_to_string_and_offset(data=DATA)
DATA: List[Dict] = process_data(data=DATA, label_to_query=LABEL_TO_QUERY)
JSON_WRITER.write(x=DATA, file_name="data/clean_data/valid.json")

DATA: List[Dict] = read_data_from_file(file_name="data/raw_data/test.txt")
DATA: List[Dict] = convert_data_to_string_and_offset(data=DATA)
DATA: List[Dict] = process_data(data=DATA, label_to_query=LABEL_TO_QUERY)
JSON_WRITER.write(x=DATA, file_name="data/clean_data/test.json")
