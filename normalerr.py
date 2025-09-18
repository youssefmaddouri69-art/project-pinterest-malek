import torch


def collate_fn(batch):
    images, paths, success_flags = zip(*batch)
    valid_images = [img for img, flag in zip(images, success_flags) if flag]
    if valid_images:
        return torch.stack(valid_images)
    else:
        return torch.empty(0, 3, 224, 224)
