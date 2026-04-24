"""Shared helpers for fault-detection analysis scripts."""

from unittest.mock import patch

import torch
import torch.nn.functional as F
from nafmamba.utils.channel_utils import build_channel_slices


def _mock_stty_size(command, mode="r"):
    if command.strip().startswith("stty size"):
        return type("DummyPopen", (), {"read": lambda self: "24 80"})()
    return _ORIGINAL_OS_POPEN(command, mode)


def load_tester_rs():
    import os

    global _ORIGINAL_OS_POPEN
    _ORIGINAL_OS_POPEN = os.popen
    with patch("os.popen", new=_mock_stty_size):
        from nafmamba.utils import TesterRS

    return TesterRS


def find_contaminated_weights(random_tensors, fault_probability):
    contaminated_layers = []
    dim0 = []
    dim1 = []
    dim2 = []
    dim3 = []

    threshold = 1 - fault_probability
    for layer_idx, tensor in enumerate(random_tensors):
        indices = torch.nonzero(tensor > threshold, as_tuple=False)
        for coord in indices:
            coord_list = coord.tolist()
            contaminated_layers.append(layer_idx)
            dim0.append(coord_list[0] if len(coord_list) > 0 else None)
            dim1.append(coord_list[1] if len(coord_list) > 1 else None)
            dim2.append(coord_list[2] if len(coord_list) > 2 else None)
            dim3.append(coord_list[3] if len(coord_list) > 3 else None)

    return contaminated_layers, dim0, dim1, dim2, dim3

def run_router_with_channel_splits(noisy, denoisers, router_module, model_channels=31):
    total_channels = noisy.size(1)
    channel_slices, remainder = build_channel_slices(
        total_channels, model_channels=model_channels
    )

    outputs = []
    for start, end in channel_slices:
        noisy_chunk = noisy[:, start:end, :, :]
        chunk_features = [denoiser(noisy_chunk) for denoiser in denoisers]
        outputs.append(noisy_chunk - router_module(chunk_features))

    if remainder == 0:
        return torch.cat(outputs, dim=1)

    return torch.cat(outputs[:-1] + [outputs[-1][:, -remainder:, :, :]], dim=1)


def get_router_pred_with_channel_splits(
    noisy,
    denoisers,
    router_module,
    model_channels=31,
    resize_to=None,
):
    total_channels = noisy.size(1)
    channel_slices, _ = build_channel_slices(total_channels, model_channels=model_channels)
    chunk_preds = []

    for start, end in channel_slices:
        noisy_chunk = noisy[:, start:end, :, :]
        if resize_to is not None:
            noisy_chunk = F.interpolate(
                noisy_chunk,
                size=resize_to,
                mode="bilinear",
                align_corners=False,
            )

        chunk_features = [denoiser(noisy_chunk) for denoiser in denoisers]
        chunk_preds.append(router_module(chunk_features).float())

    return (torch.stack(chunk_preds).min(dim=0).values >= 0.5).float()
