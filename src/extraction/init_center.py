import torch
from tqdm import tqdm
import numpy as np


def kmeans_init(features_data):
    print("In the process of initializing the center")

    data = torch.tensor(features_data, device='cuda')

    n = len(data)
    sqrt_n = int(n ** 0.5)
    centers = []

    while len(centers) < sqrt_n:
        sse_min = float('inf')
        b_unit = tqdm(enumerate(range(n)), total=n)

        for i in range(n):
            center = centers.copy()
            if len(center) < 1 or not torch.any(torch.all(data[i] == torch.stack(center), dim=1)):
                center.append(data[i])
                center = torch.stack(center)
                sse = 0.0

                cluster_labels = torch.argmin(torch.cdist(data, center), dim=1)

                for j in range(len(center)):
                    cluster_points = data[cluster_labels == j]
                    single_sse = torch.sum(torch.norm(cluster_points - center[j], dim=1))
                    sse += single_sse.item()

                if sse < sse_min:
                    sse_min = sse
                    join_center = data[i]
                    label = cluster_labels.clone()

                if i % 10000 == 0:
                    b_unit.set_description(f"kmeans_init {sqrt_n}: {len(centers)}, {len(data)} - sse_min: {sse_min}, "
                                           f" sse: {sse} - center[0].shape: {center[0].shape}, device: {center[0].device}")
                    b_unit.refresh()

            if i % 10000 == 0:
                b_unit.update(1)

        centers.append(join_center)
        b_unit.n = n
        b_unit.refresh()
        b_unit.close()

    centers_cpu = [tensor.cpu() for tensor in centers]
    centers_np = [tensor.numpy() for tensor in centers_cpu]
    del centers
    del centers_cpu

    label_cpu = [tensor.cpu() for tensor in label]
    label_np = [tensor.numpy() for tensor in label_cpu]
    del label
    del label_cpu
    return np.array(label_np), np.array(centers_np)
