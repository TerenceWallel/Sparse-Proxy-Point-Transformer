import torch

def get_obb2world(points):
    cov = torch.cov(points.T)
    _, _, eigen_vectors = torch.svd(cov.float())
    return eigen_vectors / torch.norm(eigen_vectors, dim=1, keepdim=True, p=2)
