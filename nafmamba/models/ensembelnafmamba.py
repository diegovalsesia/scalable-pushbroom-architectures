from hydra.utils import to_absolute_path
import logging
import torch
import torch.nn as nn
import numpy as np
import time
from nafmamba.models.losses import PSNRLoss, STDLoss, ScalableLoss
from .metrics import mpsnr, mse, psnr
from .base import BaseModel
import nafmamba.models.layers as layers
from nafmamba.models.layers.nafmamba_feature import Model
from nafmamba.models.layers.dynamicrouter import EfficientDynamicRouter
from .utils import PatchesHandler
import random
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EnsembleNAFMambaModel(BaseModel):
    def __init__(
        self,
        base,
        num_denoisers,
        channels,
        features,
        ssl=0,
        n_ssl=0,
        sample_stage_lambda=0.0,
        ckpt=None,
        ckpt_denoisers=None,
    ):
        super().__init__(**base)
        self.num_denoisers = num_denoisers
        self.sample_stage_lambda = sample_stage_lambda
        self.channels = channels
        self.feature = features
        self.layers_params = layers
        self.ssl = ssl
        self.n_ssl = n_ssl
        logger.debug(f"ssl : {self.ssl}, n_ssl : {self.n_ssl}")
        self.normalized_dict = False
        # self.ensmble = nn.ModuleList([ Model(bands= self.channels,feature = self.feature) for _ in range(num_denoisers)])
        self.ensemble = nn.ModuleList([Model(bands=self.channels,feature=self.feature) for _ in range(num_denoisers)])
        logger.info(f"Using SSL : {self.ssl}")
        self.ckpt = ckpt
        self.router = EfficientDynamicRouter(in_dim=self.feature, out_dim=self.channels)
        # self.router = SimpleDynamicRouter(in_dim=self.feature, out_dim=self.channels)
        if self.ckpt is not None:
            logger.info(f"Loading ckpt {self.ckpt!r}")
            d = torch.load(to_absolute_path(self.ckpt))
            self.load_state_dict(d["state_dict"])
        elif ckpt_denoisers is not None:
            if len(ckpt_denoisers) != num_denoisers:
                raise ValueError("Number of ckpt_denoisers must be equal to num_denoisers")
            for i, ckpt in enumerate(ckpt_denoisers):
                if ckpt is not None:
                    logger.info(f"Loading ckpt {ckpt!r}")
                    d = torch.load(to_absolute_path(ckpt))
                    sub_state = {}
                    for key, value in d['state_dict'].items():
                        if key.startswith('net.'):
                            new_key = key[len('net.'):]
                            sub_state[new_key] = value
                    sub_state.keys()
                    self.ensemble[i].load_state_dict(sub_state)
                    logger.info(f"Loaded ckpt {i}")
            # for param in self.ensemble.parameters():
            #     param.requires_grad = False


    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        y = batch.pop("y")
        

        out = self.forward(**batch,is_train=True)
        self.log("train_mse", mse(out, y))
        self.log("train_psnr", psnr(out, y))
        self.log("train_mpsnr", mpsnr(out.detach(), y))

        loss = mse(out, y)
        # loss = self.uncertainty_loss(out, out_features, y)
        self.log("train_loss", loss)      
        self.manual_backward(loss)
        opt.step()
        sch = self.lr_schedulers()
        if self.trainer.is_last_batch:
            epoch = self.current_epoch
            lr = sch.get_last_lr()
            logger.info(f"Epoch {epoch} : lr={lr} \t loss={loss:.6f}")
            sch.step()

    def validation_step(self, batch, batch_idx):
        y = batch.pop("y")
        start = time.time()
        if self.block_inference and self.block_inference.use_bi:
            out,outputs = self.forward_blocks(**batch,is_train=False)
        else:
            out,outputs = self.forward(**batch, is_train=False)
        logger.debug(f"Val denoised shape: {out.shape}")
        out = out.clamp(0, 1)
        outputs = [o.clamp(0, 1) for o in outputs]
        elapsed = time.time() - start
        _mse = mse(out, y)
        _mpsnr = mpsnr(out, y)
        logger.debug(f"Val mse : {_mse}, mpsnr: {_mpsnr}")
        self.log("val_mse", mse(out, y))
        self.log("val_psnr", psnr(out, y))
        for i,o in enumerate(outputs):
            self.log(f"val_mpsnr_denoiser{i}", mpsnr(o, y))
        del outputs
        self.log("val_mpsnr", mpsnr(out, y))
        self.log("val_batch_time", elapsed)
        self.log("val_psnr_noise", psnr(batch["x"], y))
        self.log("val_mpsnr_noise", mpsnr(batch["x"], y))

    def test_step(self, batch, batch_idx):
        y = batch.pop("y")

        if self.block_inference and self.block_inference.use_bi:
            out,ouputs = self.forward_blocks(**batch)
        else:
            out,ouputs = self.forward(**batch, is_train=False)
        logger.debug(f"Test denoised shape: {out.shape}")
        out = out.clamp(0, 1)
        outputs = [o.clamp(0, 1) for o in outputs]
        for i,o in enumerate(outputs):
            self.log(f"test_mpsnr_denoiser{i}", mpsnr(o, y))
        del outputs
        self.log("test_mse", mse(out, y))
        self.log("test_psnr", psnr(out, y))
        self.log("test_mpsnr", mpsnr(out, y))
        self.log("test_psnr_noise", psnr(batch["x"], y))
        self.log("test_mpsnr_noise", mpsnr(batch["x"], y))
    def forward(
        self, x, is_train=False,inference=False, index_initialize_states=0, mode=None, img_id=None, sigmas=None, ssl_idx=None, **kwargs
    ):
        sample_stage = self.sample_stage(self.num_denoisers, self.sample_stage_lambda)
        # sample_stage = torch.randint(1, self.num_denoisers + 1, (1,)).item()
        num_nets = sample_stage if is_train else self.num_denoisers
        streams = [torch.cuda.Stream() for _ in range(num_nets)]
        outputs = [None] * len(self.ensemble)
        outputs_features = [None] * num_nets
        if is_train:
            # nets = self.ensemble[:sample_stage]
            nets = random.sample(list(self.ensemble), sample_stage)
        else:
            nets = self.ensemble
        for i, net in enumerate(nets):
            with torch.cuda.stream(streams[i]):
                outputs_features[i] = net(x,inference=inference, index_initialize_states=index_initialize_states)
        torch.cuda.synchronize()
        avg_out = x - self.router(outputs_features) ## net is used to predict the noise
        # avg_out = self.router(outputs_features[:num_nets])
        if is_train:
            return avg_out
        else:
            outputs = [x - self.router(outputs_features[:i+1]) for i in range(self.num_denoisers)]
            # outputs = [self.router(outputs_features[:i+1]) for i in range(self.num_denoisers)]
        # x = torch.stack(outputs,dim=0).mean(dim=0)

        return avg_out, outputs


    def forward_blocks(self, x, **kwargs):
        logger.debug(f"Starting block inference")
        # device_ = x.device
        # x = x.to('cpu')
        # print(x.device)
        block_size = min(
            max(x.shape[-1], x.shape[-2]), self.block_inference.block_size
        )
        patches_handler = PatchesHandler(
            size=(block_size,) * 2,
            channels=x.shape[1],
            stride=block_size - self.block_inference.overlap,
            padding=self.block_inference.padding,
        )

        logger.debug(f"Forward patches handler")
        blocks_in = patches_handler(x, mode="extract").clone()
        blocks_grid = tuple(blocks_in.shape[-2:])
        logger.debug(f"blocks grid : {blocks_in.shape}")

        blocks_out = torch.zeros_like(blocks_in)
        blocks_outputs = [torch.zeros_like(blocks_in)  for _ in range(self.num_denoisers)]
        
        # blocks_in = blocks_in.to(device_)
        logger.debug(f"Processing blocks {blocks_grid}")
        for i in range(blocks_grid[0]):
            for j in range(blocks_grid[1]):
                print(i*blocks_grid[1]+j,'/',blocks_grid[0]*blocks_grid[1],end='\r')
                # print(blocks_in.device, blocks_out.device)
                tmp_in = blocks_in[:, :, :, :, i, j]

                blocks_ij, outputs_ij = self.forward(tmp_in, **kwargs)
                blocks_ij= blocks_ij
                blocks_out[:, :, :, :, i, j] = blocks_ij
                for k, o in enumerate(outputs_ij):
                    blocks_outputs[k][ :, :, :, :, i, j] = o
        logger.debug(f"Blocks processed")
        x = patches_handler(blocks_out, mode="aggregate")
        for i in range(len(blocks_outputs)):
            blocks_outputs[i] = patches_handler(blocks_outputs[i], mode="aggregate")
        logger.debug(f"Blocks aggregated to shape : {tuple(x.shape)}")
        # return x.to(device_)
        return x, blocks_outputs
    
    @staticmethod
    def sample_stage(num_denoisers, sample_stage_lambda):
        prob = torch.exp(sample_stage_lambda * torch.arange(1, num_denoisers+1))
        return torch.multinomial(prob, 1).item() + 1
