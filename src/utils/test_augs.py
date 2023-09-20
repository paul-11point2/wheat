import logging
import warnings

import hydra
import torch
import random
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import omegaconf
import numpy as np

from src.utils.get_dataset import get_training_datasets, load_replay_augs
from src.utils.transforms import ResizeWithinBounds
from src.utils.visualise import visualise

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

def get_augs():
    transform = A.ReplayCompose([
        ResizeWithinBounds(max_height=2160, max_width=4096, interpolation=1, p=1.0),
        A.PadIfNeeded(min_height=2160, min_width=4096, position='random', border_mode=0, value=0.0, p=1.0),
        A.RandomBrightnessContrast(p=0.6),
        A.pytorch.transforms.ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    return transform

          
def test(cfg: omegaconf.DictConfig) -> None:
    train_datasets = get_training_datasets(cfg)
    train = train_datasets['train']
    # remove transforms so that we can control transform in test
    train.transforms = None
    
    replay_transforms = get_augs() # load_replay_augs(cfg['augmentation']['train'])
    random.seed(7)

    while True:
        try:
            idx = input('Enter image index: ')
            img, target, image_id = train.__getitem__(int(idx))
            target['bboxes'] = target['boxes']
            visualise('Original', img, target)
            orig_dict = {
                'image': img,
                'image_id': image_id,
                'bboxes'
                : target['boxes'],
                'labels': target['labels']
            }

            transformed_dict = replay_transforms(**orig_dict)
            visualize('Transformed', image, transformed_dict)
            logger.info(transformed_dict['replay'])
        except Exception as e:
            logger.exception(f'Couldnt get image {idx}')
    

# Note that config_name must be provided on the command line. ie.
# python hydra_run --config-name=wheat
@hydra.main(config_path='../../conf')
def run_test(cfg: omegaconf.DictConfig) -> None:
    if len(cfg) == 0:
        raise ValueError(f'Hydra configuration is empty. Looked in configuration directory "./conf". Did you specify --config-name=<CONF_FILE> on the command line?')
    logger.info(omegaconf.OmegaConf.to_yaml(cfg))
    test(cfg)
    input('Press <ENTER> to continue')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_test()
