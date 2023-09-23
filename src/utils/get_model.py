from pathlib import Path
from typing import Any

from omegaconf import DictConfig
import torch

from src.utils.utils import load_obj

# def _default_anchorgen():
#     anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
#     aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
#     return AnchorGenerator(anchor_sizes, aspect_ratios)

def get_wheat_model(cfg: DictConfig) -> Any:
    """
    Get model

    Args:
        cfg: config

    Returns:
        initialized model
    """

    # anchor_generator = load_obj(cfg.model.anchor_generator.class_name)
    # anchor_sizes = cfg.model.anchor_generator.params.anchor_sizes
    # aspect_ratios = [cfg.model.anchor_generator.params.aspect_ratios,] * len(anchor_sizes)
    # anchor_generator = anchor_generator(anchor_sizes, aspect_ratios)
    
    model = load_obj(cfg.model.backbone.class_name)
    # model = model(rpn_anchor_generator=anchor_generator, **cfg.model.backbone.params)
    model = model(**cfg.model.backbone.params)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    head = load_obj(cfg.model.head.class_name)

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = head(in_features, cfg.model.head.params.num_classes)

    if cfg.model.get('state_dict') and cfg.model.state_dict:
        weights_checkpoint_path = Path(cfg.general.checkpoint_dir, cfg.model.state_dict).expanduser()
        state_dict = torch.load(weights_checkpoint_path)['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        model.load_state_dict(state_dict)

    return model
