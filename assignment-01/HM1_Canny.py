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

    # Quantize gradient direction to 4 main directions: 0, 45, 90, 135 degrees.
    angle = (np.rad2deg(grad_dir) + 180.0) % 180.0

    h, w = grad_mag.shape
    padded = np.zeros((h + 2, w + 2), dtype=grad_mag.dtype)
    padded[1:-1, 1:-1] = grad_mag

    center = padded[1:-1, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    up_left = padded[:-2, :-2]
    up_right = padded[:-2, 2:]
    down_left = padded[2:, :-2]
    down_right = padded[2:, 2:]

    mask_0 = (angle < 22.5) | (angle >= 157.5)
    mask_45 = (angle >= 22.5) & (angle < 67.5)
    mask_90 = (angle >= 67.5) & (angle < 112.5)
    mask_135 = (angle >= 112.5) & (angle < 157.5)

    keep_0 = mask_0 & (center >= left) & (center >= right)
    keep_45 = mask_45 & (center >= up_right) & (center >= down_left)
    keep_90 = mask_90 & (center >= up) & (center >= down)
    keep_135 = mask_135 & (center >= up_left) & (center >= down_right)

    keep = keep_0 | keep_45 | keep_90 | keep_135
    NMS_output = np.where(keep, center, 0.0)
    return NMS_output 
            


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """
    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.1
    high_ratio = 0.3

    mean_val = np.mean(img)
    high_threshold = mean_val * high_ratio
    low_threshold = mean_val * low_ratio

    strong = img >= high_threshold
    weak = (img >= low_threshold) & (~strong)

    # Iteratively link weak edges that are 8-connected to strong edges.
    while True:
        padded = np.zeros((strong.shape[0] + 2, strong.shape[1] + 2), dtype=bool)
        padded[1:-1, 1:-1] = strong

        connected = (
            padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
            padded[1:-1, :-2] |                     padded[1:-1, 2:] |
            padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:]
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
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
