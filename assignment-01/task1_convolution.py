import numpy as np


def pad2d(img, pad, mode="zero"):
    """
    2D padding without numpy.pad.
    mode: 'zero' or 'replicate'.
    """
    if pad == 0:
        return img.copy()

    h, w = img.shape
    if mode == "zero":
        out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=img.dtype)
        out[pad:pad + h, pad:pad + w] = img
        return out

    if mode == "replicate":
        # Use clipped index grids to replicate boundary pixels.
        y = np.clip(np.arange(-pad, h + pad), 0, h - 1)
        x = np.clip(np.arange(-pad, w + pad), 0, w - 1)
        return img[y[:, None], x[None, :]]

    raise ValueError("mode must be 'zero' or 'replicate'")


def build_doubly_block_toeplitz_6x6_k3(kernel):
    """
    Build DBT matrix A for 6x6 input and 3x3 kernel (valid convolution):
        vec(Y_4x4) = A @ vec(X_6x6)
    """
    if kernel.shape != (3, 3):
        raise ValueError("kernel must be 3x3")

    h, w = 6, 6
    kh, kw = 3, 3
    oh, ow = h - kh + 1, w - kw + 1

    oy, ox = np.meshgrid(np.arange(oh), np.arange(ow), indexing="ij")
    ky, kx = np.meshgrid(np.arange(kh), np.arange(kw), indexing="ij")

    # Advanced indexing creates all 3x3 receptive-field linear indices at once.
    rows = (oy[..., None, None] + ky).reshape(oh * ow, -1)
    cols = (ox[..., None, None] + kx).reshape(oh * ow, -1)
    lin = rows * w + cols

    a = np.zeros((oh * ow, h * w), dtype=kernel.dtype)
    k_flat = kernel[::-1, ::-1].reshape(1, -1)
    a[np.arange(oh * ow)[:, None], lin] = k_flat
    return a


def conv2d_6x6_via_dbt(x6, k3):
    """Single matmul valid convolution for 6x6 input and 3x3 kernel."""
    if x6.shape != (6, 6) or k3.shape != (3, 3):
        raise ValueError("x6 must be (6, 6) and k3 must be (3, 3)")
    a = build_doubly_block_toeplitz_6x6_k3(k3)
    y = a @ x6.reshape(-1)
    return y.reshape(4, 4)


def im2col_indices(h, w, kh, kw, stride=1):
    """
    Generate im2col sampling indices via meshgrid (no loops).
    """
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1

    oy, ox = np.meshgrid(np.arange(oh), np.arange(ow), indexing="ij")
    ky, kx = np.meshgrid(np.arange(kh), np.arange(kw), indexing="ij")

    iy = oy[..., None, None] * stride + ky
    ix = ox[..., None, None] * stride + kx
    return iy, ix, oh, ow


def conv2d_im2col(img, kernel, stride=1, padding=0, pad_mode="zero"):
    """
    Generic single-channel 2D convolution with im2col + one matrix multiplication.
    """
    kh, kw = kernel.shape
    x = pad2d(img, padding, mode=pad_mode)

    iy, ix, oh, ow = im2col_indices(x.shape[0], x.shape[1], kh, kw, stride)
    cols = x[iy, ix].reshape(oh * ow, kh * kw)
    k = kernel[::-1, ::-1].reshape(-1, 1)

    out = cols @ k
    return out.reshape(oh, ow)
