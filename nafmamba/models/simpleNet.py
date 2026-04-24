from hydra.utils import to_absolute_path
import logging
import torch
from .base import BaseModel
import nafmamba.models.layers as layers
from nafmamba.models.layers.simplenet import SimpleNet as SimpleNetarch
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SimpleNet(BaseModel):
    def __init__(self,base,in_nc=4, out_nc=4, nf=32,
        ssl=0,
        n_ssl=0,ckpt=None,):
        super().__init__(**base)
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.nf = nf
        self.layers_params = layers
        self.ssl = ssl
        self.n_ssl = n_ssl
        logger.debug(f"ssl : {self.ssl}, n_ssl : {self.n_ssl}")
        self.net = SimpleNetarch(in_nc=in_nc, out_nc=out_nc, nf=nf)
        # self.init_layers()
        logger.info(f"Using SSL : {self.ssl}")
        self.ckpt = ckpt
        if self.ckpt is not None:
            try:
                logger.info(f"Loading ckpt {self.ckpt!r}")
                d = torch.load(to_absolute_path(self.ckpt))
                self.load_state_dict(d["state_dict"])
            except:
                print("Could not load ckpt")
                pass

    def forward(self,x, mode=None, img_id=None, sigmas=None, ssl_idx=None, **kwargs
    ):
        x = self.net(x)
        return x
