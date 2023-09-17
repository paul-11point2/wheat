import os
import warnings

import hydra
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
import torch
import omegaconf
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger

from src.lightning_classes.wheat_data_module import WheatDataModule
from src.lightning_classes.lightning_wheat import LitWheat
# from src.utils.loggers import JsonLogger
from src.utils.utils import set_seed, save_useful_info, flatten_omegaconf

warnings.filterwarnings("ignore")


def run(cfg: omegaconf.DictConfig) -> None:
    """
    Run pytorch-lightning model

    Args:
        cfg: hydra config

    """
    set_seed(cfg.training.seed)
    hparams = flatten_omegaconf(cfg)

    data_module = WheatDataModule(cfg=cfg)
    model = LitWheat(datamodule=data_module, hparams=hparams, cfg=cfg)
    
    # early_stopping = EarlyStopping(**cfg.callbacks.early_stopping.params)
    model_checkpoint = ModelCheckpoint(**cfg.callbacks.model_checkpoint.params)
    strategy = DDPStrategy(find_unused_parameters=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    tb_logger = TensorBoardLogger(save_dir=cfg.general.save_dir, version=cfg.logging.version)
    # comet_logger = CometLogger(save_dir=cfg.general.save_dir,
    #                            workspace=cfg.general.workspace,
    #                            project_name=cfg.general.project_name,
    #                            api_key=cfg.private.comet_api,
    #                            experiment_name=os.getcwd().split('\\')[-1])
    # json_logger = JsonLogger()

    trainer = pl.Trainer(
        logger=[tb_logger],
        strategy=strategy,
        callbacks=[lr_monitor, model_checkpoint],
        **cfg.trainer,
    )
    trainer.fit(model=model, datamodule=data_module)

    # save as a simple torch model
    model_name = os.getcwd().split('\\')[-1] + '.pth'
    print(model_name)
    torch.save(model.model.state_dict(), model_name)


@hydra.main(config_path='conf', config_name='config')
def run_model(cfg: omegaconf.DictConfig) -> None:
    print(omegaconf.OmegaConf.to_yaml(cfg))
    save_useful_info()
    run(cfg)


if __name__ == '__main__':
    run_model()
