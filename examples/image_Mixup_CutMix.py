import numpy as np
from mxnet import nd

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
