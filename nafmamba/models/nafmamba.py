from hydra.utils import to_absolute_path
import logging
import torch
import torch.nn as nn

from .base import BaseModel
import nafmamba.models.layers as layers
from nafmamba.models.layers.nafmamba2 import Model
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NAFMambaModel(BaseModel):
    def __init__(
        self,
        base,
        channels,
        features,
        ssl=0,
        n_ssl=0,
        ckpt=None,
    ):
        super().__init__(**base)
        self.channels = channels
        self.feature = features
        self.layers_params = layers
        self.ssl = ssl
        self.n_ssl = n_ssl
        logger.debug(f"ssl : {self.ssl}, n_ssl : {self.n_ssl}")
        self.normalized_dict = False
        self.net = Model(bands= self.channels,feature = self.feature          )
        logger.info(f"Using SSL : {self.ssl}")
        self.ckpt = ckpt
        if self.ckpt is not None:
            logger.info(f"Loading ckpt {self.ckpt!r}")
            d = torch.load(to_absolute_path(self.ckpt))
            self.load_state_dict(d["state_dict"])

    def forward(
        self, x, mode=None, img_id=None, sigmas=None, ssl_idx=None, **kwargs
    ):
       x = self.net(x)
       return x
