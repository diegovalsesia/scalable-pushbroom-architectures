import logging

import torch
from pytorch_lightning import seed_everything

from nafmamba import models
from nafmamba.data import DataModule
from nafmamba.utils import Tester

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device in use : {device}")

    # Fix seed for reproducibility
    logger.info(f"Using random seed {cfg.seed}")
    seed_everything(cfg.seed)

    # Load requested dataloader
    datamodule = DataModule(idx_test=cfg.test.idx_test, **cfg.data.params)
    datamodule.setup(stage="test")

    model_class = models.__dict__[cfg.model.class_name]
    model = model_class(**cfg.model.params)

    model = model.to(device)
    if "test_dir" in cfg.test:
        from nafmamba.utils import TesterRS as Tester
        print("Using RS Tester")
    else:
        from nafmamba.utils import Tester
    tester = Tester(**cfg.test)
    tester.eval(model, datamodule)
