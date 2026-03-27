import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad * x_grad + y_grad * y_grad)
    direction_grad = np.arctan2(y_grad, x_grad)

    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   

    h, w = grad_mag.shape

    # Bilinear interpolation sampling on zero-padded magnitude map.
    padded = np.zeros((h + 2, w + 2), dtype=grad_mag.dtype)
    padded[1:-1, 1:-1] = grad_mag

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    dx = np.cos(grad_dir).astype(np.float32)
    dy = np.sin(grad_dir).astype(np.float32)

    def bilinear_sample(yq, xq):
        y = yq + 1.0
        x = xq + 1.0

        y0 = np.floor(y).astype(np.int32)
        x0 = np.floor(x).astype(np.int32)
        y1 = y0 + 1
        x1 = x0 + 1

        y0 = np.clip(y0, 0, h + 1)
        x0 = np.clip(x0, 0, w + 1)
        y1 = np.clip(y1, 0, h + 1)
        x1 = np.clip(x1, 0, w + 1)

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        return wa * padded[y0, x0] + wb * padded[y0, x1] + wc * padded[y1, x0] + wd * padded[y1, x1]

    forward = bilinear_sample(yy + dy, xx + dx)
    backward = bilinear_sample(yy - dy, xx - dx)

    keep = (grad_mag >= forward) & (grad_mag >= backward)
    NMS_output = np.where(keep, grad_mag, 0.0)
    return NMS_output 
            


def hysteresis_thresholding(img, grad_dir) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """
    #you can adjust the parameters to fit your own implementation 
    low_ratio = 1.2
    high_ratio = 6.3

    max_val = np.mean(img)
    high_threshold = max_val * high_ratio
    low_threshold = max_val * low_ratio

    strong = img >= high_threshold
    weak = (img >= low_threshold) & (~strong)

    # Use gradient direction to define edge tangent direction for linking.
    angle = (np.rad2deg(grad_dir) + 180.0) % 180.0
    mask_0 = (angle < 22.5) | (angle >= 157.5)
    mask_45 = (angle >= 22.5) & (angle < 67.5)
    mask_90 = (angle >= 67.5) & (angle < 112.5)
    mask_135 = (angle >= 112.5) & (angle < 157.5)

    # Iteratively link weak edges with directional neighbors along edge tangent.
    while True:
        padded = np.zeros((strong.shape[0] + 2, strong.shape[1] + 2), dtype=bool)
        padded[1:-1, 1:-1] = strong

        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        up_left = padded[:-2, :-2]
        up_right = padded[:-2, 2:]
        down_left = padded[2:, :-2]
        down_right = padded[2:, 2:]

        connected = (
            (mask_0 & (up | down)) |
            (mask_45 & (up_left | down_right)) |
            (mask_90 & (left | right)) |
            (mask_135 & (up_right | down_left))
        )

        new_strong = weak & connected & (~strong)
        if not np.any(new_strong):
            break
        strong = strong | new_strong

    output = strong.astype(np.float32)
    return output 



if __name__=="__main__":

    #Load the input images
    input_img = read_img("Lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output, direction_grad)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
