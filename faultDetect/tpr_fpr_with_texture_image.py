import copy
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from faultDetect.corehyi import fault_injection
from faultDetect.common import (
    find_contaminated_weights,
    get_router_pred_with_channel_splits,
    load_tester_rs,
)
from nafmamba.data.datasets.mat_dataset import MatImageDatasetwithName
from nafmamba.models.ensembelnafmamba import EnsembleNAFMambaModel
from nafmamba.models.layers.dynamicrouter import LightweightAttention_singlehead
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
Tester = load_tester_rs()
device = "cuda:1"


tester = Tester(name='',save_labels=False,save_raw=False,save_rgb=False,save_rgb_crop=False,seed=0,idx_test= "",
test_dir="your_path",
gt_dir= None)
dataset = MatImageDatasetwithName(tester.test_dir,tester.gt_dir)
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
class EfficientDynamicRouter_returnidx(nn.Module):
    def __init__(self, in_dim, out_dim,drop_rate=0.1,use_drop=False):
        super().__init__()
        self.attn = LightweightAttention_singlehead_drop_return_idx(in_dim, drop_rate=drop_rate)
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
        correct_idx = self.attn(features)  # [BHW, C]
        return correct_idx


class LightweightAttention_singlehead_drop_return_idx(LightweightAttention_singlehead):
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
        correct_idx = self.drop_and_reweight_compare_diagonal_threshold(attn, threshold=threshold)
        return correct_idx

    def drop_and_reweight_compare_diagonal_threshold(self, attention, threshold=0.01):
        attn_diagonal = attention.diagonal(dim1=1, dim2=2)  # shape: (BHW, N)
        atten_diagonal_variance = attn_diagonal.var(dim=0)  # shape: (N,)
        # keep_diag = attention_sum_diagonal >= threshold * BHW
        keep_diag = atten_diagonal_variance <= threshold
        # if not keep_diag.any():
        #     min_idx = atten_diagonal_variance.argmin()
        #     keep_diag = torch.zeros_like(keep_diag)
        #     keep_diag[min_idx] = True
        ## return the indices of the kept diagonals, 1 means keep, 0 means drop
        return torch.where(keep_diag, torch.ones_like(keep_diag), torch.zeros_like(keep_diag))

droprouter = EfficientDynamicRouter_returnidx(in_dim=96,out_dim=31,use_drop=True)
droprouter = droprouter.to(device)
droprouter.load_state_dict(router.state_dict())
texture_img_name_list =['Labtest_0910-1506','Lehavim_0910-1622','Lehavim_0910-1626','Lehavim_0910-1627','Lehavim_0910-1636','Lehavim_0910-1716','Lehavim_0910-1716','gavyam_0823-0950-1','nachal_0823-1149','nachal_0823-1220','negev_0823-1005','rmt_0328-1249-1']
optimizer = torch.optim.Adam(droprouter.parameters(), lr=1e-4)
fault_possibilities = [ 1e-8, 5e-8, 1e-7,5e-7, 1e-6, 5e-6, 1e-5]
tpr_counts = []
fpr_counts = []
texture_tpr_counts = []
texture_fpr_counts = []
for fault_possibility in fault_possibilities:
    tpr_counts_for_each_possibility = []
    fpr_counts_for_each_possibility = []
    texture_tpr_counts_for_each_possibility = []
    texture_fpr_counts_for_each_possibility = []
    for i in range(30):
        tp_counts_for_each_run = []
        fp_counts_for_each_run = []
        tn_counts_for_each_run = []
        fn_counts_for_each_run = []
        texture_tp_counts_for_each_run = []
        texture_fp_counts_for_each_run = []
        texture_tn_counts_for_each_run = []
        texture_fn_counts_for_each_run = []
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
            name = data['name']
            # if gt.sum() == 0:
            #     best_output = noisy
            #     dropout = noisy
            #     normal_output = noisy
            # else:
            with torch.no_grad():
                pred = get_router_pred_with_channel_splits(
                    noisy=noisy,
                    denoisers=new_ensemble,
                    router_module=droprouter,
                    model_channels=31,
                    resize_to=(512, 512),
                )
            True_positive_mask = (pred == 0) & (gt == 0)
            False_positive_mask = (pred == 0) & (gt == 1)
            True_negative_mask = (pred == 1) & (gt == 1)
            False_negative_mask = (pred == 1) & (gt == 0)
            true_positive = True_positive_mask.sum().item()
            false_positive = False_positive_mask.sum().item()
            true_negative = True_negative_mask.sum().item()
            false_negative = False_negative_mask.sum().item()
            tp_counts_for_each_run.append(true_positive)
            fp_counts_for_each_run.append(false_positive)
            tn_counts_for_each_run.append(true_negative)
            fn_counts_for_each_run.append(false_negative)
            if name[0] in texture_img_name_list:
                texture_tp_counts_for_each_run.append(true_positive)
                texture_fp_counts_for_each_run.append(false_positive)
                texture_tn_counts_for_each_run.append(true_negative)
                texture_fn_counts_for_each_run.append(false_negative)
            tpr_for_each_run = sum(tp_counts_for_each_run) / (sum(tp_counts_for_each_run) + sum(fn_counts_for_each_run)) if (sum(tp_counts_for_each_run) + sum(fn_counts_for_each_run)) > 0 else 0
            fpr_for_each_run = sum(fp_counts_for_each_run) / (sum(fp_counts_for_each_run) + sum(tn_counts_for_each_run)) if (sum(fp_counts_for_each_run) + sum(tn_counts_for_each_run)) > 0 else 0
            texture_tpr_for_each_run = sum(texture_tp_counts_for_each_run) / (sum(texture_tp_counts_for_each_run) + sum(texture_fn_counts_for_each_run)) if (sum(texture_tp_counts_for_each_run) + sum(texture_fn_counts_for_each_run)) > 0 else 0
            texture_fpr_for_each_run = sum(texture_fp_counts_for_each_run) / (sum(texture_fp_counts_for_each_run) + sum(texture_tn_counts_for_each_run)) if (sum(texture_fp_counts_for_each_run) + sum(texture_tn_counts_for_each_run)) > 0 else 0
            print(f"Fault Possibility: {fault_possibility}, TPR: {tpr_for_each_run:.4f}, FPR: {fpr_for_each_run:.4f}, Texture TPR: {texture_tpr_for_each_run:.4f}, Texture FPR: {texture_fpr_for_each_run:.4f}")
        tpr_counts_for_each_possibility.append(tpr_for_each_run)
        fpr_counts_for_each_possibility.append(fpr_for_each_run)
        texture_tpr_counts_for_each_possibility.append(texture_tpr_for_each_run)
        texture_fpr_counts_for_each_possibility.append(texture_fpr_for_each_run)
    tpr_counts.append(sum(tpr_counts_for_each_possibility) / len(tpr_counts_for_each_possibility))
    fpr_counts.append(sum(fpr_counts_for_each_possibility) / len(fpr_counts_for_each_possibility))
    texture_tpr_counts.append(sum(texture_tpr_counts_for_each_possibility) / len(texture_tpr_counts_for_each_possibility))
    texture_fpr_counts.append(sum(texture_fpr_counts_for_each_possibility) / len(texture_fpr_counts_for_each_possibility))
results = {
    "fault_possibilities": fault_possibilities,
    "tpr_counts": tpr_counts,
    "fpr_counts": fpr_counts,
    "texture_tpr_counts": texture_tpr_counts,
    "texture_fpr_counts": texture_fpr_counts
}
with open('fault_injection_results_pavia.json', 'w') as f:
    json.dump(results, f)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(fault_possibilities, tpr_counts, marker='o', label='Overall TPR')
plt.plot(fault_possibilities, fpr_counts, marker='o', label='Overall FPR')
# plt.plot(fault_possibilities, texture_tpr_counts, marker='o', label='Texture TPR')
# plt.plot(fault_possibilities, texture_fpr_counts, marker='o', label='Texture FPR')
plt.xscale('log')
plt.xlabel('Fault Possibility (log scale)')
plt.ylabel('Rate')
plt.title('TPR and FPR vs Fault Possibility')
plt.legend()
plt.grid(True)
plt.savefig('tpr_fpr_vs_fault_possibility_pavia.pdf', dpi=300, bbox_inches='tight')
plt.show()
