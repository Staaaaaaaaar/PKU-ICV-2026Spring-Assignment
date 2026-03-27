import numpy as np


def required_ransac_iters(w_est, sample_size=3, success_prob=0.999):
    """
    Minimum iterations for success probability p:
        N >= log(1-p) / log(1-w^s)
    """
    w = np.clip(w_est, 1e-6, 1.0 - 1e-6)
    denom = np.log(1.0 - (w ** sample_size))
    n_iters = int(np.ceil(np.log(1.0 - success_prob) / denom))
    return max(n_iters, 1)


def fit_plane_ransac_parallel(points, dist_thresh, w_est=0.5, success_prob=0.999, rng_seed=0):
    """
    Parallel RANSAC plane fitting for 3D points.

    Steps:
    1) Pre-compute required iterations from target success rate.
    2) Generate all hypotheses in parallel (vectorized random triplets).
    3) Evaluate all point-to-plane distances with broadcasting.
    4) Pick best inlier set and refine with least squares (SVD).

    Returns:
        plane: [a, b, c, d] where ax + by + cz + d = 0 and ||[a,b,c]||=1
        inlier_mask: shape (N,)
        n_iters: used iteration count
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 3:
        raise ValueError("points must have shape (N, 3), N >= 3")

    n_points = p.shape[0]
    n_iters = required_ransac_iters(w_est, sample_size=3, success_prob=success_prob)
    rng = np.random.default_rng(rng_seed)

    # Vectorized triplet sampling with index-fixing to reduce duplicates in each row.
    idx = rng.integers(0, n_points, size=(n_iters, 3), endpoint=False)

    same01 = idx[:, 1] == idx[:, 0]
    idx[same01, 1] = (idx[same01, 1] + 1) % n_points

    same02 = idx[:, 2] == idx[:, 0]
    idx[same02, 2] = (idx[same02, 2] + 2) % n_points

    same12 = idx[:, 2] == idx[:, 1]
    idx[same12, 2] = (idx[same12, 2] + 1) % n_points

    same20 = idx[:, 2] == idx[:, 0]
    idx[same20, 2] = (idx[same20, 2] + 1) % n_points

    a = p[idx[:, 0]]
    b = p[idx[:, 1]]
    c = p[idx[:, 2]]

    normals = np.cross(b - a, c - a)
    norm_len = np.linalg.norm(normals, axis=1)
    valid = norm_len > 1e-12

    normals_unit = np.zeros_like(normals)
    normals_unit[valid] = normals[valid] / norm_len[valid, None]
    d = -np.sum(normals_unit * a, axis=1)

    # Broadcasting computes all distances at once: (N, T).
    dist = np.abs(p @ normals_unit.T + d[None, :])
    inliers = dist < dist_thresh
    counts = inliers.sum(axis=0)
    counts = np.where(valid, counts, -1)

    best_idx = int(np.argmax(counts))
    best_mask = inliers[:, best_idx]

    pin = p[best_mask]
    if pin.shape[0] < 3:
        plane = np.concatenate([normals_unit[best_idx], np.array([d[best_idx]])])
        return plane, best_mask, n_iters

    centroid = pin.mean(axis=0)
    x = pin - centroid
    _, _, vt = np.linalg.svd(x, full_matrices=False)

    n = vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    d_refined = -np.dot(n, centroid)

    plane = np.concatenate([n, np.array([d_refined])])
    return plane, best_mask, n_iters
