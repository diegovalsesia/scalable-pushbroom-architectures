### this code is used to statistics the varaince of the ensemble model outputs under fault injection
import copy
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from faultDetect.corehyi import fault_injection
from faultDetect.common import find_contaminated_weights, load_tester_rs
from nafmamba.data.datasets.mat_dataset import MatImageDataset
from nafmamba.models.ensembelnafmamba import EnsembleNAFMambaModel
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm

##### These class is for router that return varaince of the outputs

class LightweightAttention_singlehead(nn.Module):
    def __init__(self, dim, drop_rate=0.1):
        super().__init__()
        self.scale = dim ** -0.5
        
        self.proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.pos_emb = nn.Parameter(torch.randn(1, 1, dim))
        self.drop_rate = drop_rate

    def forward(self, x):
        bhw, n, f = x.shape
        x = x + self.pos_emb  # add positional embedding
        
        # generate queries, keys, values
        qkv = self.proj(x).chunk(3, dim=-1)  # q, k, v: (bhw, n, dim)
        q, k, v = qkv
        
        # attention: (bhw, n, n)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # print("attn matrix in first pixel",attn[0,:,:])
        # out: (bhw, n, dim)
        out = torch.matmul(attn, v)
        # print("out matrix in first pixel",out[0,:,:])
        projected_out = self.out_proj(out)
        # print(x[0,:,:])
        # print("projected_out matrix in first pixel",projected_out[0,:,:])
        result = (projected_out + x).mean(dim=1)
        return result
    
class LightweightAttention_singlehead_drop(LightweightAttention_singlehead):
    def forward(self, x, threshold=0.01):
        bhw, n, f = x.shape
        x = x + self.pos_emb  # add positional embedding
        
        # generate queries, keys, values
        qkv = self.proj(x).chunk(3, dim=-1)  # q, k, v: (bhw, n, dim)
        q, k, v = qkv
        # attention: (bhw, n, n)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # print("attn matrix in temp pixel",attn[0,:,:])
        attn, variance = self.drop_and_reweight_compare_diagonal_threshold(attn, threshold=threshold)
        out = torch.matmul(attn, v)
        keep_nozero = out.sum(dim=(0, 2)) != 0
        # print("keep_nozero",keep_nozero)
        x_val = x[:,keep_nozero,:]
        # print("x_val",x_val[0,:,:])
        # print(x_val.shape)
        out = out[:,keep_nozero,:]
        out = self.out_proj(out)
        # print("out",out[0,:,:])
        # result =(out +x_val).sum(dim=1)/keep_rows.sum(dim=1).unsqueeze(-1)
        result = (out + x_val).mean(dim=1)
        return result, variance
            

    
    def drop_and_reweight_compare_diagonal_threshold(self, attention, threshold=0.01):
        BHW, N, C = attention.shape
        attn_diagonal = attention.diagonal(dim1=1, dim2=2)  # shape: (BHW, N)
        atten_diagonal_variance = attn_diagonal.var(dim=0)  # shape: (N,)
        # keep_diag = attention_sum_diagonal >= threshold * BHW
        keep_diag = atten_diagonal_variance <= threshold
        if not keep_diag.any():
            min_idx = atten_diagonal_variance.argmin()
            keep_diag = torch.zeros_like(keep_diag)
            keep_diag[min_idx] = True
        keep_mask = keep_diag.unsqueeze(1).expand(N, C)
        keep_mask = keep_mask& keep_mask.transpose(0, 1)
        keep_mask = keep_mask.unsqueeze(0).expand(BHW, N, C)  # shape: (BHW, N, C)
        # print('threshold',threshold)

        atten_masked = attention * keep_mask.float()
        atten_masked_for_sm = atten_masked.clone()
        # atten_masked_for_sm[atten_masked_for_sm == 0] = float('-inf')
        # attention = F.softmax(atten_masked_for_sm, dim=2)
        atten_masked_for_sm = atten_masked_for_sm / (atten_masked_for_sm.sum(dim=2, keepdim=True) + 1e-6)
        attention = torch.nan_to_num(atten_masked_for_sm, nan=0.0)
        return attention,atten_diagonal_variance                     # row‐wise softmax  

    
class EfficientDynamicRouter(nn.Module):
    def __init__(self, in_dim, out_dim,drop_rate=0.1,use_drop=False):
        super().__init__()
        if not use_drop:
            self.attn = LightweightAttention_singlehead(in_dim, drop_rate=drop_rate)
        else: 
            self.attn = LightweightAttention_singlehead_drop(in_dim, drop_rate=drop_rate)
        self.conv_finale = nn.Conv1d(in_channels= in_dim , out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(in_dim)
        

    def forward(self, denoiser_outputs):
        """

        """
        features = torch.stack(denoiser_outputs, dim=1)  # [B, N, F, H, W]
        B, N, C, H, W = features.shape
        features = rearrange(features, 'b n c h w -> b h w n c')
        features = rearrange(features, 'b h w n c-> (b h w) n c')
        features = self.norm(features)
        # calculate  weights output
        weighted, variance = self.attn(features)  # [BHW, C]
        # recover spatial information
        # spatial = rearrange(weighted, '(b h w) c -> b c h w', b=B, h=H, w=W)
        weighted_reshape = rearrange(weighted, " (B H W) C-> B H W C", B=B, H=H, W=W, C=C)
        weighted_reshape = rearrange(weighted_reshape, " B H W C-> (B H) C W ", B=B, H=H, W=W, C=C)
        out = self.conv_finale(weighted_reshape)
        out = rearrange(out, "(B H) C W -> B C H W", B=B, H=H)
        return out, variance


Tester = load_tester_rs()
device = "cuda:1"

tester = Tester(name='',save_labels=False,save_raw=False,save_rgb=False,save_rgb_crop=False,seed=0,idx_test= "",
test_dir="your_path",
gt_dir= "your_path")
dataset = MatImageDataset(tester.test_dir,tester.gt_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)   
base = {
    'optimizer': None,
    'lr_scheduler': None,
    'block_inference': {
        'use_bi': True,
        'block_size': 512,
        'overlap': 0,
        'padding': 'reflect'
    }
}
number_of_denoisers = 5
model = EnsembleNAFMambaModel(base,channels=31, features=96, num_denoisers=number_of_denoisers,ckpt='your_path')
model = model.to(device)
ensemble =copy.deepcopy(model.ensemble)
ensemble = ensemble.to(device)
router = copy.deepcopy(model.router)
router = router.to(device)
del model
droprouter = EfficientDynamicRouter(in_dim=96,out_dim=31,use_drop=True)
droprouter = droprouter.to(device)
droprouter.load_state_dict(router.state_dict())

optimizer = torch.optim.Adam(droprouter.parameters(), lr=1e-4)
fault_possibilties = [1e-7,5e-7,1e-6]
correct_variance_list_all_possibility = {}
fault_variance_list_all_possibility = {}
for fault_possibility in fault_possibilties:
    correct_variance_list = []
    fault_variance_list = []
    for i in range(10):
        oracle_psnr_epoch = []
        threshold_psnr_epoch = []
        faulty_psnr_epoch = []
        new_ensemble = copy.deepcopy(ensemble).to(device)
        droprouter.eval()
        pfi_model = fault_injection(model=ensemble[0],batch_size=1,input_shape=[31,512,512],layer_types = [nn.Conv1d,nn.Linear], is_cuda=True)
        random_indices = [pfi_model.layer_weights_size_random() for _ in range(number_of_denoisers)]
        del pfi_model
        gt = torch.zeros(number_of_denoisers).to(device)
        for i,random_indice in enumerate(random_indices):
            contaminated_layers, dim0, dim1, dim2, dim3 = find_contaminated_weights(random_indice, fault_possibility)
            if len(contaminated_layers) == 0:
                gt[i] = 1
            else:
                pfi_model = fault_injection(model=new_ensemble[i],batch_size=1,input_shape=[31, 512,512],layer_types = [nn.Conv1d,nn.Linear], is_cuda=True)
                fi = pfi_model.declare_weight_fi(layer_num=contaminated_layers,k=dim0, dim1=dim1,dim2=dim2, dim3=dim3,value=[random.randint(1000, 100000) for _ in range(len(contaminated_layers))])
                new_ensemble[i] = fi
        for i, data in enumerate(tqdm(dataloader)):     
            noisy = data['x'].to(device)
            clean = data['y'].to(device)
            features =[]
            with torch.no_grad():
                for j in range(number_of_denoisers):
                    features.append(new_ensemble[j](noisy))
                noisy_pred, variance = droprouter(features)
                for idx, var in enumerate(variance):
                    if gt[idx] == 1:
                        correct_variance_list.append(var.cpu().numpy())
                    else:
                        fault_variance_list.append(var.cpu().numpy())
    correct_variance_list_all_possibility[fault_possibility] = correct_variance_list
    fault_variance_list_all_possibility[fault_possibility] = fault_variance_list
for key in correct_variance_list_all_possibility.keys():
    print(f"Fault Possibility: {key}")
    print(f"Correct Denoisers Variance Count: {len(correct_variance_list_all_possibility[key])}")
    print(f"Faulty Denoisers Variance Count: {len(fault_variance_list_all_possibility[key])}")
with open('correct_variance_list_all_possibility_testforrelease.pkl', 'wb') as f:
    pickle.dump(correct_variance_list_all_possibility, f)

with open('fault_variance_list_all_possibility_testforrelease.pkl', 'wb') as f:
    pickle.dump(fault_variance_list_all_possibility, f)

print("Saved variance dictionaries to disk.")

fig, axes = plt.subplots(
    len(fault_possibilties),
    1,
    figsize=(10, 4 * len(fault_possibilties)),
    squeeze=False,
)

for axis, fault_possibility in zip(axes.flatten(), fault_possibilties):
    correct_values = np.asarray(
        correct_variance_list_all_possibility[fault_possibility]
    ).reshape(-1)
    faulty_values = np.asarray(
        fault_variance_list_all_possibility[fault_possibility]
    ).reshape(-1)

    if correct_values.size > 0:
        axis.hist(
            correct_values,
            bins=50,
            alpha=0.6,
            label="Correct denoisers",
            density=True,
        )
    if faulty_values.size > 0:
        axis.hist(
            faulty_values,
            bins=50,
            alpha=0.6,
            label="Faulty denoisers",
            density=True,
        )

    axis.set_title(f"Fault probability = {fault_possibility}")
    axis.set_xlabel("Attention diagonal variance")
    axis.set_ylabel("Density")
    axis.grid(True, alpha=0.3)
    axis.legend()

fig.tight_layout()
fig.savefig(
    "fault_variance_histogram_testforrelease.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("Saved variance histogram to fault_variance_histogram_testforrelease.pdf.")

    
