import copy
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
from faultDetect.common import (
    find_contaminated_weights,
    load_tester_rs,
    run_router_with_channel_splits,
)
from nafmamba.data.datasets.mat_dataset import MatImageDataset
from nafmamba.models.ensembelnafmamba import EnsembleNAFMambaModel
from nafmamba.models.metrics.metrics import mpsnr
from torch.utils.data import DataLoader
from tqdm import tqdm
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
from nafmamba.models.layers.dynamicrouter import EfficientDynamicRouter
droprouter = EfficientDynamicRouter(in_dim=96,out_dim=31,use_drop=True)
droprouter = droprouter.to(device)
droprouter.load_state_dict(router.state_dict())

optimizer = torch.optim.Adam(droprouter.parameters(), lr=1e-4)
fault_possibilities = [ 1e-8, 5e-8, 1e-7,5e-7, 1e-6, 5e-6, 1e-5]
threshold_psnr = []
faulty_psnr = []
oracle_psnr = []
for fault_possibility in fault_possibilities:
    oracle_psnr_each_possibility = []
    threshold_psnr_each_possibility = []
    faulty_psnr_each_possibility = []
    for i in range(30):
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
            if gt.sum() == 0:
                best_output = noisy
                dropout = noisy
                normal_output = noisy
            else:
                with torch.no_grad():
                    dropout = run_router_with_channel_splits(
                        noisy=noisy,
                        denoisers=new_ensemble,
                        router_module=droprouter,
                        model_channels=31,
                    )

                    normal_output = run_router_with_channel_splits(
                        noisy=noisy,
                        denoisers=new_ensemble,
                        router_module=router,
                        model_channels=31,
                    )

                    indices = gt.nonzero(as_tuple=True)[0].tolist()
                    best_denoisers = [new_ensemble[k] for k in indices]
                    best_output = run_router_with_channel_splits(
                        noisy=noisy,
                        denoisers=best_denoisers,
                        router_module=router,
                        model_channels=31,
                    )
            oracle_psnr_epoch.append(mpsnr(clean, best_output))
            threshold_psnr_epoch.append(mpsnr(clean, dropout))
            faulty_psnr_epoch.append(mpsnr(clean, normal_output))
        print("fault_possibility: ",fault_possibility)
        print("oracle_psnr: ",sum(oracle_psnr_epoch)/len(oracle_psnr_epoch))
        print("threshold_psnr: ",sum(threshold_psnr_epoch)/len(threshold_psnr_epoch))
        print("faulty_psnr: ",sum(faulty_psnr_epoch)/len(faulty_psnr_epoch))
        oracle_psnr_each_possibility.append(sum(oracle_psnr_epoch)/len(oracle_psnr_epoch))
        threshold_psnr_each_possibility.append(sum(threshold_psnr_epoch)/len(threshold_psnr_epoch))
        faulty_psnr_each_possibility.append(sum(faulty_psnr_epoch)/len(faulty_psnr_epoch))
    oracle_psnr.append(sum(oracle_psnr_each_possibility)/len(oracle_psnr_each_possibility))
    threshold_psnr.append(sum(threshold_psnr_each_possibility)/len(threshold_psnr_each_possibility))
    faulty_psnr.append(sum(faulty_psnr_each_possibility)/len(faulty_psnr_each_possibility))
print("oracle_psnr: ",oracle_psnr)
print("threshold_psnr: ",threshold_psnr)
print("faulty_psnr: ",faulty_psnr)
print('gap between oracle and threshold: ', [o - f for o, f in zip(oracle_psnr, threshold_psnr)])
log_possibility = np.log10(fault_possibilities)
plt.plot(log_possibility, [30.65]*len(log_possibility), label='original_psnr', linestyle='--')
plt.plot(log_possibility, oracle_psnr, label='oracle_psnr')
plt.plot(log_possibility, threshold_psnr, label='filtered_psnr')
plt.plot(log_possibility, faulty_psnr, label='faulty_psnr')
plt.xticks(ticks=log_possibility, labels=['1e-8', '5e-8', '1e-7', '5e-7', '1e-6', '5e-6', '1e-5'])
plt.xlabel('Fault Possibility')
plt.ylabel('PSNR')
plt.title('Fault Injection Analysis')
plt.legend()
plt.grid()
plt.show()
plt.savefig('fault_inject_HSI_testforrealse.pdf', dpi=300, bbox_inches='tight')
plt.close()
