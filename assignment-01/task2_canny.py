import numpy as np
from collections import deque


def pad2d(img, pad, mode="zero"):
    if pad == 0:
        return img.copy()

    h, w = img.shape
    if mode == "zero":
        out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=img.dtype)
        out[pad:pad + h, pad:pad + w] = img
        return out

    if mode == "replicate":
        y = np.clip(np.arange(-pad, h + pad), 0, h - 1)
        x = np.clip(np.arange(-pad, w + pad), 0, w - 1)
        return img[y[:, None], x[None, :]]

    raise ValueError("mode must be 'zero' or 'replicate'")


def im2col_indices(h, w, kh, kw, stride=1):
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1

    oy, ox = np.meshgrid(np.arange(oh), np.arange(ow), indexing="ij")
    ky, kx = np.meshgrid(np.arange(kh), np.arange(kw), indexing="ij")

    iy = oy[..., None, None] * stride + ky
    ix = ox[..., None, None] * stride + kx
    return iy, ix, oh, ow


def conv2d_im2col(img, kernel, stride=1, padding=0, pad_mode="zero"):
    kh, kw = kernel.shape
    x = pad2d(img, padding, mode=pad_mode)

    iy, ix, oh, ow = im2col_indices(x.shape[0], x.shape[1], kh, kw, stride)
    cols = x[iy, ix].reshape(oh * ow, kh * kw)
    k = kernel[::-1, ::-1].reshape(-1, 1)

    out = cols @ k
    return out.reshape(oh, ow)


def sobel_gradients(img):
    """
    Compute Ix, Iy, gradient magnitude M and direction D.
    """
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float64)

    ix = conv2d_im2col(img, kx, stride=1, padding=1, pad_mode="replicate")
    iy = conv2d_im2col(img, ky, stride=1, padding=1, pad_mode="replicate")

    m = np.hypot(ix, iy)
    d = np.arctan2(iy, ix)
    return ix, iy, m, d


def non_maximum_suppression(magnitude, direction):
    """
    Loop-free NMS using quantized orientation and advanced indexing.
    """
    h, w = magnitude.shape
    deg = (np.rad2deg(direction) + 180.0) % 180.0
    bins = ((deg + 22.5) // 45).astype(np.int32) % 4

    offsets = np.array([
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
    ], dtype=np.int32)

    dy = offsets[bins, 0]
    dx = offsets[bins, 1]

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    mp = pad2d(magnitude, 1, mode="zero")

    cy = y + 1
    cx = x + 1

    m1 = mp[cy + dy, cx + dx]
    m2 = mp[cy - dy, cx - dx]

    keep = (magnitude >= m1) & (magnitude >= m2)
    return magnitude * keep


def hysteresis_edge_linking(nms, low, high):
    """
    Double-threshold edge linking.
    Loops are intentionally used here (explicitly allowed by assignment).
    """
    strong = nms >= high
    weak = (nms >= low) & (nms < high)

    h, w = nms.shape
    edges = np.zeros((h, w), dtype=np.uint8)
    edges[strong] = 1

    q = deque(np.argwhere(strong).tolist())
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    while q:
        yy, xx = q.popleft()
        for dy, dx in neighbors:
            ny, nx = yy + dy, xx + dx
            if 0 <= ny < h and 0 <= nx < w and weak[ny, nx] and edges[ny, nx] == 0:
                edges[ny, nx] = 1
                q.append((ny, nx))

    return edges


def canny_pipeline(img, low, high):
    """Convenience wrapper for Canny core stages."""
    _, _, m, d = sobel_gradients(img)
    nms = non_maximum_suppression(m, d)
    edges = hysteresis_edge_linking(nms, low=low, high=high)
    return m, d, nms, edges
