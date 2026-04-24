import os
import glob
import scipy.io
import torch
from torch.utils.data import Dataset
from hydra.utils import to_absolute_path
from nafmamba.data.normalizers import GlobalMinMax

def load_mat_image(filepath, key='x'):
    mat = scipy.io.loadmat(filepath)
    img = mat['DataCube']  
    tensor_img = torch.tensor(img, dtype=torch.float32)
    tensor_img = tensor_img.permute(2, 0, 1)
    return tensor_img


class MatImageDataset(Dataset):
    def __init__(self, noisy_path, gt_path=None):
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_path, "*.mat")))
        self.gt_files = sorted(glob.glob(os.path.join(gt_path, "*.mat"))) if gt_path is not None else []
        if gt_path is not None:
            assert len(self.noisy_files) == len(self.gt_files), "length_mismatch"
        self.normalizer = GlobalMinMax()
    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        noisy_filepath = self.noisy_files[idx]
        gt_filepath = self.gt_files[idx] if self.gt_files else None
        noisy_img = load_mat_image(noisy_filepath)
        if gt_filepath is not None:
            gt_img = load_mat_image(gt_filepath)
            gt_img = self.normalizer.transform(gt_img).clone()
        else:
            gt_img = noisy_img.clone()
        return {"x": noisy_img, "y": gt_img}


class MatImageDatasetwithName(MatImageDataset):
    def __getitem__(self, idx):
        noisy_filepath = self.noisy_files[idx]
        gt_filepath = self.gt_files[idx] if self.gt_files else None
        noisy_img = load_mat_image(noisy_filepath)
        if gt_filepath is not None:
            gt_img = load_mat_image(gt_filepath)
            gt_img = self.normalizer.transform(gt_img).clone()
        else:
            gt_img = noisy_img.clone()
        name = os.path.basename(noisy_filepath).split('.')[0]
        return {"x": noisy_img, "y": gt_img, "name": name}