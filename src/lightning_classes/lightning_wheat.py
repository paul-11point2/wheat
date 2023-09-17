from typing import Any, Dict, Mapping, Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils.coco_eval import CocoEvaluator
from src.utils.coco_utils import get_coco_api_from_dataset, _get_iou_types
from src.utils.get_dataset import get_training_datasets
from src.utils.get_model import get_wheat_model
from src.utils.utils import load_obj, collate_fn


class LitWheat(pl.LightningModule):
    def __init__(self, datamodule: pl.LightningDataModule, hparams: Dict[str, float], cfg: DictConfig):
        super(LitWheat, self).__init__()
        self.cfg = cfg
        self.data_module = datamodule
        self.save_hyperparameters(hparams)
        self.model = get_wheat_model(self.cfg)
        self.coco_evaluator: Optional[CocoEvaluator] = None
        
        # if hasattr(self.model, 'parameters'):
        #     self.hparams['n_params'] = sum(p.numel() for p in self.model.parameters())

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return self.model.load_state_dict(state_dict=state_dict, strict=strict)
    
    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        if 'decoder_lr' in self.cfg.optimizer.params.keys():
            params = [
                {'params': self.model.decoder.parameters(), 'lr': self.cfg.optimizer.params.lr},
                {'params': self.model.encoder.parameters(), 'lr': self.cfg.optimizer.params.decoder_lr},
            ]
            optimizer = load_obj(self.cfg.optimizer.class_name)(params)

        else:
            optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return [optimizer], [{'scheduler': scheduler,
                              'interval': self.cfg.scheduler.step,
                              'monitor': self.cfg.scheduler.monitor}]

    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        # separate losses
        loss_dict = self.model(images, targets)
        # total loss
        losses = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict, sync_dist=True, prog_bar=True)
        self.log('losses', losses, sync_dist=True, prog_bar=True)
        return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        images = list(img for img in images)
#         targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)
        outputs = [{k: v for k, v in t.items()} for t in outputs]
        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        if self.coco_evaluator:
            self.coco_evaluator.update(res)
        else:
            raise ValueError(f'self.coco_evaluator not set')

        return {}

    def on_validation_epoch_start(self):
        # re-initialise coco_evaluator at the start of each epoch
        coco = get_coco_api_from_dataset(self.data_module.valid_dataset)
        iou_types = _get_iou_types(self.model)        
        self.coco_evaluator = CocoEvaluator(coco, iou_types)
        
    def on_validation_epoch_end(self):
        if self.coco_evaluator:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            # coco main metric
            metric = self.coco_evaluator.coco_eval['bbox'].stats[0]
            metric = torch.as_tensor(metric)
            self.log('main_score', metric, sync_dist=True, prog_bar=True)
            # return {'val_loss': metric, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
        else:
            raise ValueError(f'self.coco_evaluator not set')
