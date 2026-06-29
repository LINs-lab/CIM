import torch
import torch.nn as nn
import torch.nn.functional as F
from data.utils.load_model import get_features


def cross_entropy(y_pre, y):
    y_pre = F.softmax(y_pre, dim=1)
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]


def batched_forward(model, tensor, batch_size):
    total_samples = tensor.size(0)

    all_outputs = []

    for i in range(0, total_samples, batch_size):
        batch_data = tensor[i : min(i + batch_size, total_samples)]

        output = model(batch_data)
        # output = get_features(model, batch_data, [2])[0][0]

        all_outputs.append(output)

    final_output = torch.cat(all_outputs, dim=0)

    return final_output


def selector(n, model, images, labels):
    with torch.no_grad():
        images = images.cuda()
        preds = batched_forward(model, images, images.shape[0])

        dist = cross_entropy(preds, labels.cuda())

        indices = torch.argsort(dist, descending=False)[:n]

        # images = images.cuda()
        # labels = labels.cuda()

        # images_normalized = normalize(images)

        # features = batched_forward(model, images_normalized, images.shape[0])

        # indices = []
        # indices_full = torch.arange(len(features))

        # feature_c = features
        # indices_c = indices_full

        # feature_mean = feature_c.mean(0, keepdim=True)
        # current_sum = torch.zeros_like(feature_mean)

        # cur_indices = []
        # for k in range(n):
        #     target = (k + 1) * feature_mean - current_sum
        #     dist = torch.norm(target - feature_c, dim=1)
        #     indices_sorted = torch.argsort(dist, descending=False)

        #     # We can make this faster by reducing feature matrix
        #     for idx in indices_sorted:
        #         idx = idx.item()
        #         if idx not in cur_indices:
        #             cur_indices.append(idx)
        #             break
        #     current_sum += feature_c[idx]

        # indices.append(indices_c[cur_indices])

    return images[indices].detach()


def kod(x, y, lambda_reg=0.01):
    """
    Compute Kernel Observed Discrepancy (KOD) between two batches of data.

    Parameters:
    - x: Tensor, shape (batch_size, feature_dim)
    - y: Tensor, shape (batch_size, feature_dim)

    Returns:
    - cost: Tensor, transport cost
    """

    def compute_kernel_values(x, y):
        kernel_values = torch.norm((x.unsqueeze(0) - y.unsqueeze(1)), dim=2)
        # kernel_values = (x.unsqueeze(0) * y.unsqueeze(1)).mean(-1)
        # kernel_values = torch.exp(-1 * (kernel_values))
        return kernel_values

    # Compute pairwise squared distances
    y_sq = (compute_kernel_values(y, y)).mean(1)
    xy = (compute_kernel_values(x, y)).mean(1)

    # Compute cost
    mmd = ((y_sq - xy).abs()).mean()
    cost = (1 - lambda_reg) * mmd + lambda_reg * (x.mean(0) - y.mean(0)).norm()

    return cost


def mmd(x, y, lambda_reg=0.05):
    """
    Compute Maximum Mean Discrepancy (MMD) between two batches of data.

    Parameters:
    - x: Tensor, shape (batch_size, feature_dim)
    - y: Tensor, shape (batch_size, feature_dim)

    Returns:
    - cost: Tensor, transport cost
    """

    def compute_kernel_values(x, y):
        kernel_values = torch.norm((x.unsqueeze(0) - y.unsqueeze(1)), dim=2)
        # kernel_values = x.unsqueeze(0) * y.unsqueeze(1)
        return kernel_values

    # Compute pairwise squared distances
    x_sq = (compute_kernel_values(x, x)).mean()
    y_sq = (compute_kernel_values(y, y)).mean()
    xy = (compute_kernel_values(x, y)).mean()

    # Compute cost
    mmd = (x_sq + y_sq - 2 * xy).abs()
    cost = (1 - lambda_reg) * mmd + lambda_reg * (x.mean(0) - y.mean(0)).norm()

    return cost


def dist(x, y, method="l1_mean"):
    """Distance objectives"""
    x, y = x.view(x.shape[0], -1), y.view(y.shape[0], -1)
    if method == "mse":
        dist_ = (x.mean(0) - y.mean(0)).pow(2).sum()

    elif method == "l1":
        dist_ = (x.mean(0) - y.mean(0)).abs().sum()

    elif method == "l1_mean":
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()

    elif method == "l2_mean":
        n_b = x.shape[0]
        dist_ = (x - y).pow(2).reshape(n_b, -1).mean(-1).sum()

    elif method == "cos":
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(
            1
            - torch.sum(x * y, dim=-1)
            / (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6)
        )

    elif method == "mmd":
        dist_ = mmd(x, y)

    elif method == "kod":
        dist_ = kod(x, y)

    elif method == "ho":
        dist_ = ho(x, y)

    elif method == "ins":
        # dist_ = (x - y).mean(0).pow(2).sum()
        dist_ = (x - y).mean(0).abs().sum()

    return dist_


def matchloss(feat, feat_tg):
    """Matching losses"""
    # with torch.no_grad():
    #     feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)[0]
    #     feat_tg = feat_tg.view(feat_tg.shape[0], -1)

    # feat = model.get_feature(img_syn, args.idx_from, args.idx_to)[0]
    # feat = feat.view(feat.shape[0], -1)

    loss = dist(feat, feat_tg)
    return loss


def ho(X, Y, n_kernels=5, mul_factor=2.0, bandwidth=None):
    L2_distances = torch.cdist(torch.vstack([X, Y]), torch.vstack([X, Y])) ** 2

    if bandwidth is None:
        n_samples = L2_distances.shape[0]
        bandwidth = L2_distances.data.sum() / (n_samples**2 - n_samples)

    bandwidth_multipliers = mul_factor ** (
        torch.arange(n_kernels).cuda() - n_kernels // 2
    )

    K = torch.exp(
        -L2_distances[None, ...] / (bandwidth * bandwidth_multipliers)[:, None, None]
    ).sum(dim=0)

    X_size = X.shape[0]
    XX = K[:X_size, :X_size].mean()
    XY = K[:X_size, X_size:].mean()
    YY = K[X_size:, X_size:].mean()

    return XX - 2 * XY + YY
