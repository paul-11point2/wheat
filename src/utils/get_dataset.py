import ast
import logging
from typing import Dict

import albumentations as A
import numpy as np
import omegaconf
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.utils.utils import load_obj


logger = logging.getLogger(__name__)


def load_augs(cfg: DictConfig) -> A.Compose:
    """
    Load albumentations

    Args:
        cfg:

    Returns:
        compose object
    """
    augs = []
    for a in cfg.augs:
        if a['class_name'] == 'albumentations.OneOf':
            small_augs = []
            for small_aug in a['params']:
                # yaml can't contain tuples, so we need to convert manually
                params = {k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v)) for k, v in
                          small_aug['params'].items()}
                aug = load_obj(small_aug['class_name'])(**params)
                small_augs.append(aug)
            aug = load_obj(a['class_name'])(small_augs)
            augs.append(aug)

        else:
            params = {k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v)) for k, v in
                      a['params'].items()}
            aug = load_obj(a['class_name'])(**params)
            augs.append(aug)

    return A.Compose(augs, bbox_params=A.BboxParams(format=cfg.bbox_params.format, label_fields=cfg.bbox_params.label_fields))


def load_replay_augs(cfg: DictConfig) -> A.Compose:
    """
    Load albumentations with ReplayCompose for debug purposes

    Args:
        cfg:

    Returns:
        ReplayCompose (debug) object
    """
    augs = []
    for a in cfg.augs:
        if a['class_name'] == 'albumentations.OneOf':
            small_augs = []
            for small_aug in a['params']:
                # yaml can't contain tuples, so we need to convert manually
                params = {k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v)) for k, v in
                          small_aug['params'].items()}
                aug = load_obj(small_aug['class_name'])(**params)
                small_augs.append(aug)
            aug = load_obj(a['class_name'])(small_augs)
            augs.append(aug)

        else:
            params = {k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v)) for k, v in
                      a['params'].items()}
            aug = load_obj(a['class_name'])(**params)
            augs.append(aug)

    return A.ReplayCompose(augs, bbox_params=A.BboxParams(format=cfg.bbox_params.format, label_fields=cfg.bbox_params.label_fields))


def get_training_datasets(cfg: DictConfig) -> Dict:
    """
    Get datases for modelling

    Args:
        cfg: config

    Returns:
        dict with datasets
    """

    train = pd.read_csv(f'{cfg.data.folder_path}/{cfg.data.csv_file}')


    if cfg.data.get('max_training_items') and cfg.data.max_training_items > 0:
        len_train = len(train)
        train = train[0:cfg.data.max_training_items]
        logger.info(f'WARNING: Truncating number of training items to specified cfg.data.max_training_items={cfg.data.max_training_items} from a total of {len_train}')
        

    # for fast training
    if cfg.training.debug:
        logger.info(f'WARNING: DEBUG ACTIVE. Truncating number of training items 10')
        train = train[:10]

    train[['x', 'y', 'w', 'h']] = pd.DataFrame(np.stack(train['bbox'].apply(lambda x: ast.literal_eval(x)))).astype(
        np.float32
    )

    # precalculate some values
    train['x1'] = train['x'] + train['w']
    train['y1'] = train['y'] + train['h']
    train['area'] = train['w'] * train['h']
    train_ids, valid_ids = train_test_split(train['image_id'].unique(), test_size=0.1, random_state=cfg.training.seed)

    train_df = train.loc[train['image_id'].isin(train_ids)]
    valid_df = train.loc[train['image_id'].isin(valid_ids)]

    train_img_dir = f'{cfg.data.training_img_path}'

    # train dataset
    dataset_class = load_obj(cfg.dataset.class_name)

    # initialize augmentations
    train_augs = load_augs(cfg['augmentation']['train'])
    valid_augs = load_augs(cfg['augmentation']['valid'])

    train_dataset = dataset_class(dataframe=train_df, mode='train', image_dir=train_img_dir, cfg=cfg, transforms=train_augs)

    valid_dataset = dataset_class(dataframe=valid_df, mode='valid', image_dir=train_img_dir, cfg=cfg, transforms=valid_augs)

    return {'train': train_dataset, 'valid': valid_dataset}


def get_test_dataset(cfg: DictConfig) -> object:
    """
    Get test dataset

    Args:
        cfg:

    Returns:
        test dataset
    """

    test_img_dir = f'{cfg.data.folder_path}/test'

    valid_augs = load_augs(cfg['augmentation']['valid']['augs'])
    dataset_class = load_obj(cfg.dataset.class_name)

    test_dataset = dataset_class(dataframe=None, mode='test', image_dir=test_img_dir, cfg=cfg, transforms=valid_augs)

    return test_dataset
