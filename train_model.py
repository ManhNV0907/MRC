from common.logger import SingletonLogger
from common.utils import JsonReadObjectFromLocalPatient, JsonWriteObjectToLocalPatient
from torch.utils.data import DataLoader
from typing import Dict, List
from config import (
    BATCH_SIZE, BERT_CASED_MODEL_DIR, DEVICE_NAME,
    PRETRAINED_PARAMS_LEARNING_RATE, NEW_PARAMS_LEARNING_RATE,
    STEP_SIZE, GAMMA, NUM_EPOCHS, ENTITY_TYPES_PATH
)
from datasets import NERMRCDataset, collate_func
from transformers import AutoModel, BertModel
from models import NERMRCModel
from losses import NERMRCLoss
from metrics import MRCNERMetrics
from datetime import datetime
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch import Tensor


SingletonLogger.set_logger_name(name="Train model")
LOG_DIR: str = f"log_dir/{datetime.now().strftime('%Y_%m_%d %H_%M_%S')}"
SingletonLogger.get_instance().info(f"ALL RESULT SAVING IN {LOG_DIR}")

DEVICE = torch.device(DEVICE_NAME) if torch.cuda.is_available() else torch.device("cpu")

JSON_READER = JsonReadObjectFromLocalPatient()
JSON_WRITER = JsonWriteObjectToLocalPatient()

TRAIN_DATA: List[Dict] = JSON_READER.read(file_name="data/clean_data/train.json")
TRAIN_DATASET = NERMRCDataset(data=TRAIN_DATA)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)

DEV_DATA: List[Dict] = JSON_READER.read(file_name="data/clean_data/valid.json")
DEV_DATASET = NERMRCDataset(data=DEV_DATA)
DEV_DATALOADER = DataLoader(DEV_DATASET, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_func)

TEST_DATA: List[Dict] = JSON_READER.read(file_name="data/clean_data/test.json")
TEST_DATASET = NERMRCDataset(data=TEST_DATA)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_func)

BERT_MODEL: BertModel = AutoModel.from_pretrained("bert-base-uncased", cache_dir=BERT_CASED_MODEL_DIR)
MODEL = NERMRCModel(context_presenter=BERT_MODEL, hidden_dim=768)
MODEL = MODEL.to(DEVICE)
print("MODEL ARCHITECTURE")
print(MODEL)

LOSSER = NERMRCLoss().to(DEVICE)

PRETRAINED_PARAMS: List[nn.Parameter] = []
NEW_PARAMS: List[nn.Parameter] = []
for name, param in MODEL.named_parameters():
    if "context_presenter" in name:
        PRETRAINED_PARAMS.append(param)
    else:
        NEW_PARAMS.append(param)
OPTIMIZER = Adam([
    {
        "params": PRETRAINED_PARAMS,
        "lr": PRETRAINED_PARAMS_LEARNING_RATE,
    },
    {
        "params": NEW_PARAMS,
        "lr": NEW_PARAMS_LEARNING_RATE,
    }
])
SCHEDULER = StepLR(OPTIMIZER, step_size=STEP_SIZE, gamma=GAMMA)


CONFIG: Dict = {
    "device": str(DEVICE),
    "batch_size": BATCH_SIZE,
    "pretrained_params_learning_rate": PRETRAINED_PARAMS_LEARNING_RATE,
    "new_params_learning_rate": NEW_PARAMS_LEARNING_RATE,
    "step_size": STEP_SIZE,
    "gamma": GAMMA,
    "num_epochs": NUM_EPOCHS,
    "entity_types_path": ENTITY_TYPES_PATH
}
JSON_WRITER.write(x=CONFIG, file_name=f"{LOG_DIR}/config.json")


def convert_data_dict_to_device(data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    for key, value in data_dict.items():
      if key!="label":
        data_dict[key] = value.to(DEVICE)
    return data_dict

entity_types = JSON_READER.read(CONFIG["entity_types_path"])
metrics_computer = MRCNERMetrics(
    entity_types=entity_types
)

# print(entity_types)
BEST_DEV_LOSS: float = float("inf")
GLOBAL_STEP: int = 1
for EPOCH in range(NUM_EPOCHS):
    MODEL.train()
    PROGRESS_BAR = tqdm(TRAIN_DATALOADER, desc=f"Epoch {EPOCH}: training...")
    for DATA_DICT in PROGRESS_BAR:
        # print(DATA_DICT)
        DATA_DICT: Dict[str, Tensor] = convert_data_dict_to_device(data_dict=DATA_DICT)
        START_LOGIT, END_LOGIT, MATCH_LOGIT = MODEL(
            input_ids=DATA_DICT["input_ids"], token_type_ids=DATA_DICT["token_type_ids"],
            attention_mask=DATA_DICT["attention_mask"]
        )
        LOSS = LOSSER(
            start_logit=START_LOGIT, end_logit=END_LOGIT, match_logit=MATCH_LOGIT,
            start_target=DATA_DICT["start_label"], end_target=DATA_DICT["end_label"],
            match_target=DATA_DICT["match_label"], token_type_ids=DATA_DICT["token_type_ids"]
        )
        OPTIMIZER.zero_grad()
        LOSS.backward()
        OPTIMIZER.step()
        # print(DATA_DICT["label"])
        metrics_computer.add_batch(
            START_LOGIT.detach().cpu(), 
            END_LOGIT.detach().cpu(), 
            MATCH_LOGIT.detach().cpu(), 
            DATA_DICT["start_label_mask"].detach().cpu(), 
            DATA_DICT["end_label_mask"].detach().cpu(), 
            DATA_DICT["match_label"].detach().cpu(), 
            DATA_DICT["label"],
        )
        with open(f"{LOG_DIR}/train_sum_up.txt", mode="a") as FILE_OBJ:
            FILE_OBJ.write(f"Step {GLOBAL_STEP}: loss {LOSS.item():.3f} \n")
        GLOBAL_STEP += 1
        del DATA_DICT, LOSS
        torch.cuda.empty_cache()
    PROGRESS_BAR.close()
    SCHEDULER.step()
    train_metrics = metrics_computer.compute()
    print(train_metrics["overall"])

    MODEL.eval()
    DEV_LOSSES: List[float] = []
    with torch.no_grad():
        PROGRESS_BAR = tqdm(DEV_DATALOADER, desc=f"Epoch {EPOCH}: validation...")
        for DATA_DICT in PROGRESS_BAR:
            DATA_DICT: Dict[str, Tensor] = convert_data_dict_to_device(data_dict=DATA_DICT)
            START_LOGIT, END_LOGIT, MATCH_LOGIT = MODEL(
                input_ids=DATA_DICT["input_ids"], token_type_ids=DATA_DICT["token_type_ids"],
                attention_mask=DATA_DICT["attention_mask"]
            )
            LOSS = LOSSER(
                start_logit=START_LOGIT, end_logit=END_LOGIT, match_logit=MATCH_LOGIT,
                start_target=DATA_DICT["start_label"], end_target=DATA_DICT["end_label"],
                match_target=DATA_DICT["match_label"], token_type_ids=DATA_DICT["token_type_ids"]
            )
            DEV_LOSSES.append(LOSS.item())
            metrics_computer.add_batch(
                START_LOGIT.detach().cpu(), 
                END_LOGIT.detach().cpu(), 
                MATCH_LOGIT.detach().cpu(), 
                DATA_DICT["start_label_mask"].detach().cpu(), 
                DATA_DICT["end_label_mask"].detach().cpu(), 
                DATA_DICT["match_label"].detach().cpu(), 
                DATA_DICT["label"],
            )
            del DATA_DICT, LOSS
            torch.cuda.empty_cache()
        eval_metrics = metrics_computer.compute()
        print(eval_metrics["overall"])
        PROGRESS_BAR.close()
    DEV_LOSS: float = sum(DEV_LOSSES) / len(DEV_LOSSES)

    with open(f"{LOG_DIR}/dev_sum_up.txt", mode="a") as FILE_OBJ:
        FILE_OBJ.write(f"Epoch {EPOCH}: loss {DEV_LOSS:.3f} \n")

    if DEV_LOSS < BEST_DEV_LOSS:
        print(f"LOSS IMPROVE FROM {BEST_DEV_LOSS:.3f} TO {DEV_LOSS:.3f}. SAVE MODEL")
        BEST_DEV_LOSS = DEV_LOSS
        torch.save(MODEL.state_dict(), f"{LOG_DIR}/model.pt")
checkpoint = torch.load(f"{LOG_DIR}/model.pt")
MODEL.load_state_dict(checkpoint)
MODEL.eval()
with torch.no_grad():
    PROGRESS_BAR = tqdm(TEST_DATALOADER, desc=" TEST...")
    for DATA_DICT in PROGRESS_BAR:
        DATA_DICT: Dict[str, Tensor] = convert_data_dict_to_device(data_dict=DATA_DICT)
        START_LOGIT, END_LOGIT, MATCH_LOGIT = MODEL(
            input_ids=DATA_DICT["input_ids"], token_type_ids=DATA_DICT["token_type_ids"],
            attention_mask=DATA_DICT["attention_mask"]
        )
        metrics_computer.add_batch(
            START_LOGIT.detach().cpu(), 
            END_LOGIT.detach().cpu(), 
            MATCH_LOGIT.detach().cpu(), 
            DATA_DICT["start_label_mask"].detach().cpu(), 
            DATA_DICT["end_label_mask"].detach().cpu(), 
            DATA_DICT["match_label"].detach().cpu(), 
            DATA_DICT["label"],
    )
    PROGRESS_BAR.close()

    torch.cuda.empty_cache()
    test_metrics = metrics_computer.compute()
    print(test_metrics["overall"])
    with open(f"{LOG_DIR}/test_sum_up.txt", mode="a") as FILE_OBJ:
        FILE_OBJ.write(f"Test metrics {test_metrics} \n")