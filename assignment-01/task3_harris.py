import numpy as np


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
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float64)

    ix = conv2d_im2col(img, kx, stride=1, padding=1, pad_mode="replicate")
    iy = conv2d_im2col(img, ky, stride=1, padding=1, pad_mode="replicate")
    return ix, iy


def harris_response(img, window_size=3, k=0.04):
    """
    Harris corner response:
      R = det(M) - k * tr(M)^2
    Window integration is done by box-filter convolution via im2col matmul.
    """
    ix, iy = sobel_gradients(img)

    ixx = ix * ix
    iyy = iy * iy
    ixy = ix * iy

    box = np.ones((window_size, window_size), dtype=np.float64)
    pad = window_size // 2

    sxx = conv2d_im2col(ixx, box, stride=1, padding=pad, pad_mode="replicate")
    syy = conv2d_im2col(iyy, box, stride=1, padding=pad, pad_mode="replicate")
    sxy = conv2d_im2col(ixy, box, stride=1, padding=pad, pad_mode="replicate")

    det_m = sxx * syy - sxy * sxy
    tr_m = sxx + syy
    r = det_m - k * (tr_m ** 2)
    return r
