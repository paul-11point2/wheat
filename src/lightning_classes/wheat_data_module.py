from typing import Callable, Optional
from omegaconf import DictConfig
import torch
import pandas as pd
import pytorch_lightning as pl

from src.utils.get_dataset import get_training_datasets
from src.utils.utils import  collate_fn
from src.utils.coco_utils import get_coco_api_from_dataset, _get_iou_types
from src.utils.get_dataset import get_training_datasets
from src.utils.get_model import get_wheat_model
from src.utils.utils import load_obj, collate_fn
from src.utils.coco_eval import CocoEvaluator

class WheatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(pl.LightningDataModule, self).__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        datasets = get_training_datasets(self.cfg)
        self.train_dataset = datasets['train']
        self.valid_dataset = datasets['valid']

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        raise NotImplementedError

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
        return valid_loader