from common.logger import SingletonLogger
from common.utils import JsonReadObjectFromLocalPatient
import random
from typing import Dict, List, Tuple


def display_instance(instance: Dict):
    context: str = instance['context']
    label: str = instance["label"]
    offset_mapping: List[Tuple[int, int]] = instance["offset_mapping"]
    answer_token_indexes: List[Tuple[int, int]] = instance["answer_token_indexes"]
    print(context)
    print(f"{label}:   ", end="")
    for start_token_index, end_token_index in answer_token_indexes:
        start_offset: int = offset_mapping[start_token_index][0]
        end_offset: int = offset_mapping[end_token_index][1]
        print(f"{context[start_offset:end_offset]}, ", end="")
    print("\n=========================================\n", end="")


SingletonLogger.set_logger_name(name="Inspect dataset")

JSON_READER = JsonReadObjectFromLocalPatient()
DATA: List[Dict] = JSON_READER.read(file_name="data/clean_data/train.json")
DATA: List[Dict] = list(filter(lambda x: len(x['answer_token_indexes']) > 0, DATA))
for _ in range(50):
    display_instance(instance=random.choice(DATA))
