from common.logger import SingletonLogger
from common.utils import JsonReadObjectFromLocalPatient
from transformers import AutoModel, BertModel
from models import NERMRCModel
from typing import Dict, List, Tuple
from config import PREDICT_DIRECTORY, BERT_CASED_MODEL_DIR, BERT_CASED_TOKENIZER_DIR
import torch
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from predictors import NERMRCPredictor


SingletonLogger.set_logger_name(name="Predict")

JSON_READER = JsonReadObjectFromLocalPatient()
CONFIG: Dict = JSON_READER.read(file_name=f"{PREDICT_DIRECTORY}/config.json")

DEVICE = torch.device(CONFIG["device"]) if torch.cuda.is_available() else torch.device("cpu")

TOKENIZER: BertTokenizerFast = AutoTokenizer.from_pretrained(
    "bert-base-uncased", cache_dir=BERT_CASED_TOKENIZER_DIR, use_fast=True
)

BERT_MODEL: BertModel = AutoModel.from_pretrained("bert-base-uncased", cache_dir=BERT_CASED_MODEL_DIR)
MODEL = NERMRCModel(context_presenter=BERT_MODEL, hidden_dim=768)
MODEL = MODEL.to(DEVICE)
MODEL.load_state_dict(torch.load(f"{PREDICT_DIRECTORY}/model.pt", map_location=DEVICE))
print("MODEL ARCHITECTURE")
print(MODEL)

LABEL_TO_QUERY: Dict[str, str] = JSON_READER.read(file_name="data/raw_data/label_to_query.json")

PREDICTOR = NERMRCPredictor(model=MODEL, tokenizer=TOKENIZER, device=DEVICE, label_to_query=LABEL_TO_QUERY)


def print_predict(text: str, result: Dict):
    print(f"LABEL: {result['label']}")
    print(f"ENTITIES: ", end="")
    for start_offset, end_offset in result["offsets"]:
        print(f"{text[start_offset:end_offset]}, ", end="")
    print("\n")


while True:
    print("Enter an sentence to predict, press 1 to exit")
    TEXT: str = input().strip()
    if TEXT == "1":
        break
    else:
        LIST_RESULT: List[Dict] = PREDICTOR.predict_one(text=TEXT)
        for RESULT in LIST_RESULT:
            print_predict(text=TEXT, result=RESULT)
