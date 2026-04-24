"""Utilities for channel-wise windowing and reconstruction."""


def build_channel_slices(total_channels, model_channels=31):
    if total_channels < model_channels:
        raise ValueError(
            f"Input channels ({total_channels}) must be >= model channels ({model_channels})."
        )

    full_blocks = total_channels // model_channels
    remainder = total_channels % model_channels
    channel_slices = []

    for block_idx in range(full_blocks):
        start = block_idx * model_channels
        channel_slices.append((start, start + model_channels))

    if remainder > 0:
        tail_start = total_channels - model_channels
        if not channel_slices or channel_slices[-1][0] != tail_start:
            channel_slices.append((tail_start, total_channels))

    return channel_slices, remainder
