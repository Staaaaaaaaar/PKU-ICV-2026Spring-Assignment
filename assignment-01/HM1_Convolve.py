import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    h, w = img.shape

    if type=="zeroPadding":
        padding_img = np.zeros((h + 2 * padding_size, w + 2 * padding_size), dtype=img.dtype)
        padding_img[padding_size:padding_size + h, padding_size:padding_size + w] = img
        return padding_img
    elif type=="replicatePadding":
        # build padded coordinates then clamp to nearest valid boundary by where
        row_coords = np.arange(h + 2 * padding_size) - padding_size
        col_coords = np.arange(w + 2 * padding_size) - padding_size
        rr, cc = np.meshgrid(row_coords, col_coords, indexing='ij')

        rr = np.where(rr < 0, 0, rr)
        rr = np.where(rr > h - 1, h - 1, rr)
        cc = np.where(cc < 0, 0, cc)
        cc = np.where(cc > w - 1, w - 1, cc)
        return img[rr, cc]
    else:
        raise ValueError("Unsupported padding type: {}".format(type))


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    # zero padding (for 3x3 kernel, keep output shape 6x6)
    padding_img = padding(img, 1, "zeroPadding")

    h, w = img.shape
    kh, kw = kernel.shape
    out_h = h
    out_w = w

    # Build doubly block Toeplitz matrix: (out_h*out_w) x ((h+2)*(w+2)) = 36 x 64
    toeplitz_mat = np.zeros((out_h * out_w, padding_img.size), dtype=img.dtype)

    # map each output position to 3x3 receptive field in the padded image
    r = np.arange(out_h)
    c = np.arange(out_w)
    rr, cc = np.meshgrid(r, c, indexing='ij')

    # flatten output index
    out_idx = (rr * out_w + cc).reshape(-1)

    # fill Toeplitz matrix without explicit loops
    row_offsets = np.arange(kh).reshape(1, 1, kh, 1)
    col_offsets = np.arange(kw).reshape(1, 1, 1, kw)
    col_idx = (rr.reshape(out_h, out_w, 1, 1) + row_offsets) * (w + 2) + \
              (cc.reshape(out_h, out_w, 1, 1) + col_offsets)

    col_idx = col_idx.reshape(-1, kh * kw)
    row_idx = np.repeat(out_idx, kh * kw)
    kernel_vals = np.repeat(kernel.reshape(1, -1), out_h * out_w, axis=0).reshape(-1)
    toeplitz_mat[row_idx, col_idx.reshape(-1)] = kernel_vals

    output = (toeplitz_mat @ padding_img.reshape(-1)).reshape(out_h, out_w)
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    # build im2col matrix for all sliding windows (no implicit padding)
    h, w = img.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    if out_h <= 0 or out_w <= 0:
        raise ValueError("Kernel size must be smaller than or equal to image size.")

    r = np.arange(out_h)
    c = np.arange(out_w)
    rr, cc = np.meshgrid(r, c, indexing='ij')

    row_offsets = np.arange(kh).reshape(kh, 1, 1, 1)
    col_offsets = np.arange(kw).reshape(1, kw, 1, 1)
    windows = img[rr.reshape(1, 1, out_h, out_w) + row_offsets,
                  cc.reshape(1, 1, out_h, out_w) + col_offsets]

    # shape: (kh*kw, out_h*out_w)
    im2col = windows.reshape(kh * kw, -1)
    output = (kernel.reshape(1, -1) @ im2col).reshape(out_h, out_w)

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "zeroPadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("Lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)




    