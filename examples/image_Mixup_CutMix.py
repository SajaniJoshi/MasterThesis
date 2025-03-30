import numpy as np
from mxnet import nd


"""
🧪 MixUp & CutMix Data Augmentation for MXNet

This module provides implementation of two advanced data augmentation techniques:
**MixUp** and **CutMix**, commonly used to improve model generalization and robustness.

1. mixup_data(data, labels, alpha=1.0)
   - Blends pairs of input images and labels using a convex combination.
   - Returns: mixed images, original label pairs (labels_a, labels_b), and blending factor (lam).

2. mixup_loss(loss_fn, pred, labels_a, labels_b, lam)
   - Computes the combined loss for MixUp by weighting the individual losses from both label sets.

3. cutmix_data(data, labels, alpha=1.0)
   - Replaces a random patch in each image with a patch from another image in the batch.
   - Adjusts labels proportionally based on the area of the patch.
   - Returns: modified data, label pairs, and adjusted lam value.

4. cutmix_loss(loss_fn, pred, labels_a, labels_b, lam)
   - Computes the combined loss for CutMix using the patch-based blending ratio.

📌 Note:
- `alpha` controls the Beta distribution for sampling the mixing ratio `lam`.
- `labels_a` and `labels_b` are used to interpolate ground truths.
- Designed to work with MXNet NDArrays and training pipelines.

Usage:
Apply `mixup_data` or `cutmix_data` during batch preprocessing, and use the corresponding loss function in your training loop.
"""

def mixup_data(data, labels, alpha=1.0):
    print('mixup_data')
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = data.shape[0]
    
    index = nd.array(np.random.permutation(batch_size), ctx=data.context)
    
    mixed_data = lam * data + (1 - lam) * data[index, :]
    labels_a, labels_b = labels, labels[index]
    
    return mixed_data, labels_a, labels_b, lam

def mixup_loss(loss_fn, pred, labels_a, labels_b, lam):
    return lam * loss_fn(pred, labels_a) + (1 - lam) * loss_fn(pred, labels_b)

def cutmix_data(data, labels, alpha=1.0):
    print('cutmix_data')
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size, _, H, W = data.shape
    index = nd.array(np.random.permutation(batch_size), ctx=data.context)

    # Define cutout region
    rx, ry = np.random.randint(W), np.random.randint(H)
    rw, rh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))

    x1, x2 = max(rx - rw // 2, 0), min(rx + rw // 2, W)
    y1, y2 = max(ry - rh // 2, 0), min(ry + rh // 2, H)

    # Replace region with another image
    data[:, :, y1:y2, x1:x2] = data[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    labels_a, labels_b = labels, labels[index]

    return data, labels_a, labels_b, lam

def cutmix_loss(loss_fn, pred, labels_a, labels_b, lam):
    return lam * loss_fn(pred, labels_a) + (1 - lam) * loss_fn(pred, labels_b)
